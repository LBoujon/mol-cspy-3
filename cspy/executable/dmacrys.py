from cspy.executable.executable import AbstractExecutable, ReturnCodeError
import logging
from os import remove
from os.path import exists, join
from cspy.crystal.space_group import n_symops_from_SG
from cspy.util.path import Path
from cspy.util.popen_timeout import kill
import psutil
from cspy.executable.locations import DMACRYS_EXEC
from cspy.configuration import OPT_SCHEMES, CONFIG, CspyConfiguration
import toml
from subprocess import Popen, TimeoutExpired
import re
import time
import itertools as it

CUTM_REGEX = re.compile(r"CUTM\s+[\w\.]+")
NBUR_REGEX = re.compile(r"NBUR\s+[\w\.]+")
SPLI_REGEX = re.compile(r"SPLI( [0-9]+[.][0-9]+){2}\n")

LOG = logging.getLogger("dmacrys")


class DmacrysOutputChecker:
    def __init__(
        self, output_filename, pid, period=1.0, content_timeout=10.0, kill_timeout=600.0, 
        spacegroup=None, pilot_timeout=False
    ):
        self.output_filename = output_filename
        self.timeout = content_timeout
        self.period = period
        self.pid = pid
        self.energies = []
        self.last_change = time.time()
        self.output_content = ""
        self.kill_timeout = kill_timeout
        self.error = None
        self.spacegroup = spacegroup
        self.pilot_timeout = pilot_timeout
        from threading import Thread

        self.thread = Thread(target=self.run)
        self.thread.run()

    def kj_mol_conversion(self, z, c, e):
        return (0.5 * 1389.3651171390222 / (c * z)) * e

    def check_convergence(self):
        if not self.energies:
            return True
        no_nans = all(e == e for e in self.energies[-10:])
        #good_mag = any(abs(e) < 10000 for e in self.energies[-10:])
        return no_nans# and good_mag

    def parse_energies(self):
        for line in self.output_content.splitlines():
            if "C-vector magnitude" in line:
                c = float(line.split()[2])
                LOG.debug("C-vector magnitude: %s", c)
            if "Z = " in line:
                z = float(line.split()[2])
                LOG.debug("Z: %s", z)
            if line.startswith(" Lattice energy"):
                field = line.split()[3]
                if "*" in field:
                    field = field.replace("*", "")
                    if not field:
                        e = float("nan")
                    else:
                        e = float(field)
                else:
                    e = float(field)
                self.energies.append(self.kj_mol_conversion(z, c, e))

    def update_contents(self):
        self.energies = []
        new_content = Path(self.output_filename).read_text()
        curtime = time.time()
        if new_content != self.output_content:
            self.output_content = new_content
            self.last_change = curtime

    def update_kill_timeout(self):
        # check for updates to timeout
        latest_config = CONFIG.check_for_updates()
        if latest_config.get('dmacrys.timeout_spg'):
            if str(self.spacegroup) in latest_config.get('dmacrys.timeout_spg').keys():
                old_timeout = self.kill_timeout
                self.kill_timeout = latest_config.get('dmacrys.timeout_spg')[str(self.spacegroup)]
                self.pilot_timeout = False
                LOG.debug("Dynamically updating timeout for ongoing dmacrys optimisation from %s seconds to %s seconds.", 
                          old_timeout, self.kill_timeout)


    def run(self):
        start_time = time.time()
        last_checkup = start_time
        if self.pilot_timeout:
            self.update_kill_timeout()
        while True:
            try:
                if not psutil.pid_exists(self.pid):
                    break
                time.sleep(self.period)
                self.update_contents()
                self.parse_energies()

                if not self.check_convergence():
                    self.error = "BuckCat"
                    LOG.error(
                        "Error in Dmacrys: Invalid energies encountered in output, last 10 energies: %s.\n Probable Buckingham catastrophe.",
                        self.energies[-10:],
                    )
                    kill(self.pid)
                    break
                if (time.time() - self.last_change) > self.timeout:
                    LOG.error(
                        "Output timeout in Dmacrys: no update in %s", self.timeout
                    )
                    kill(self.pid)
                    break
                if self.pilot_timeout:
                    if (time.time() - last_checkup) > 900:
                        self.update_kill_timeout()
                        last_checkup = time.time()
                if (time.time() - start_time) > self.kill_timeout:
                    if self.pilot_timeout:
                        self.error = "Timeout"
                        LOG.error("Error in Dmacrys: Timed out after dynamically updated timeout of %s seconds", self.kill_timeout)
                        kill(self.pid)
                    break
            finally:
                break


class Dmacrys(AbstractExecutable):
    _INPUT_FILE_FMT = "{}.dmain"
    _OUTPUT_FILE_FMT = "{}.dmaout"
    _executable_location = DMACRYS_EXEC
    #_timeout = CONFIG.get("dmacrys.timeout")
    _FILENAMES = {
        "star_plut": "fort.8",
        "summary": "fort.12",
        "final_cell": "fort.13",
        "fdat": "fort.15",
        "shelx": "fort.16",
        "close_contacts": "fort.17",
        "temp": "fort.30",
        "symmetry": "fort.20",
    }
    _possible_program_control_settings = (
        "IREC",
        "ACCU",
        "ACCM",
        "SCAL",
        "SYMM",
        "FDAT",
        "CHGC",
        "ZVAL",
        "EXTN",
        "NOSY"
    )
    _possible_star_settings = (
        "STAR",
        "CONP",
        "CONV",
        "PRES",
        "MAXI",
        "MAXD",
        "MAXU",
        "LIMI",
        "LIMG",
        "WCAL",
        "ENGO",
        "UDTE",
        "CCLS",
        "R2ND",
        "SEIG",
        "NOPR",
        "HESS"
    )

    def __init__(
        self,
        input_contents,
        symmetry_contents,
        working_directory=".",
        name="unknown",
        keep_files=False,
        **kwargs,
    ):
        self.name = name
        for fname in self._FILENAMES:
            setattr(self, fname + "_contents", None)
        #self._timeout = kwargs.get("timeout", self._timeout)
        self._timeout = CONFIG.get("dmacrys.timeout")
        self.input_contents = input_contents
        self.symmetry_contents = symmetry_contents
        self.output_contents = None
        self.working_directory = working_directory
        self.kwargs = kwargs.copy()
        self.energies = []
        self.adjust_cutm_nbur = kwargs.get("adjust_cutm_nbur", True)
        self.spline = kwargs.get("spline", True)
        self.error = None
        self.keep_files = keep_files
        if kwargs.get("crystal_info"):
            self.crystal_info = kwargs.get("crystal_info")
        else:
            self.crystal_info = None
        if kwargs.get("pilot_timeout"):
            self.pilot_timeout = True
        else:
            self.pilot_timeout = False

    def _parse_settings(self, string):
        lines = string.strip().split("\n")
        settings = {}
        for l in lines:
            tokens = l.split()
            if len(tokens) < 2:
                settings[tokens[0]] = True
            else:
                settings[tokens[0]] = " ".join(tokens[1:])
        return settings

    def _update_program_control_settings(self, settings):
        for k in self._possible_program_control_settings:
            if k in self.kwargs:
                settings[k] = self.kwargs[k]

        return settings
    
    def _scientific_to_standard(self, number):
        """ Convert number from scientific notation to standard notation.
         DMACRYS will not read scientific notation  """

        number = str(number).replace('E', 'e').replace('x10^', 'e')
        if 'e' in number:
            prefix, suffix = number.split('e')
            if '+' in suffix:
                exponent = int(suffix.split('+')[-1])
                number = float(prefix) * (10 ** exponent)
                number = f'{(number):.0f}'
            elif '-' in suffix:
                exponent = int(suffix.split('-')[-1])
                number = float(prefix) / (10 ** exponent)
                decimals = len(prefix.split('.')[-1])
                number = f'{(number):.{exponent + decimals}f}'

        return number

    def _update_star_settings(self, settings):
        if 'opt_scheme' in self.kwargs:
            opt_scheme = self.kwargs['opt_scheme']
            if opt_scheme in OPT_SCHEMES:
                for k in OPT_SCHEMES[opt_scheme].keys():
                    settings[k] = OPT_SCHEMES[opt_scheme][k]
            else:
                LOG.info("Error: No matching opt_scheme found. Taking defaults instead.")

        for k in self._possible_star_settings:
            if k == "CONV" and k in self.kwargs:
                settings.pop("CONP", True)
                settings["CONV"] = True
            elif k == "CONP" and k in self.kwargs:
                settings.pop("CONV", True)
                settings["CONP"] = True
            elif k == "LIMI" and k in self.kwargs:
                settings["LIMI"] = self._scientific_to_standard(self.kwargs[k])
                if not "LIMG" in self.kwargs:
                    settings["LIMG"] = self._scientific_to_standard(self.kwargs[k])
            elif k == "LIMG" and k in self.kwargs:
                settings["LIMG"] = self._scientific_to_standard(self.kwargs[k])
                if not "LIMI" in self.kwargs:
                    settings["LIMI"] = self._scientific_to_standard(self.kwargs[k])
            elif k == "UDTE" and k in self.kwargs:
                settings["UDTE"] = self._scientific_to_standard(self.kwargs[k])
            elif k == "MAXI" and k in self.kwargs:
                settings["MAXI"] = self.kwargs[k]
                if not "MAXU" in self.kwargs:
                    settings["MAXU"] = settings["MAXI"] + 1
            else:
                if k in self.kwargs:
                    settings[k] = self.kwargs[k]

        return settings

    def adjust_dmain(self):
        LOG.debug("Adjusting DMAIN program control section")
        program_control_section = ""
        program_control_regex = ""
        for start, end in it.permutations(self._possible_program_control_settings, r=2):
            regex = r"({}.*{}[^\n]*)".format(start, end)
            search = re.findall(
                regex, self.input_contents, flags=re.DOTALL | re.MULTILINE
            )
            if len(search) == 1 and len(search[0]) > len(program_control_section):
                program_control_section = search[0]
                program_control_regex = regex
        program_control_settings = self._parse_settings(program_control_section)
        self._update_program_control_settings(program_control_settings)
        lines = []
        for k, v in program_control_settings.items():
            if v is not False:
                lines.append("{} {}".format(k, "" if v is True else v).strip())
        self.input_contents = re.sub(
            program_control_regex,
            "\n".join(lines),
            self.input_contents,
            flags=re.DOTALL | re.MULTILINE,
        )

        star_section = re.findall(
            r"(STAR.*)STAR", self.input_contents, flags=re.DOTALL | re.MULTILINE
        )[0]
        star_settings = self._parse_settings(star_section)
        self._update_star_settings(star_settings)
        lines = []
        for k, v in star_settings.items():
            if v is not False:
                lines.append("{} {}".format(k, "" if v is True else v).strip())
        lines.append("STAR")
        self.input_contents = re.sub(
            r"(STAR.*)STAR",
            "\n".join(lines),
            self.input_contents,
            flags=re.DOTALL | re.MULTILINE,
        )

    def set_cutm_nbur(self, cutm=20.0, nbur=20):
        LOG.debug("Adjusting DMAIN STAR section, with: CUTM %f, NBUR %d", cutm, nbur)
        self.input_contents = re.sub(
            CUTM_REGEX, "CUTM {}".format(cutm), self.input_contents
        )
        self.input_contents = re.sub(
            NBUR_REGEX, "NBUR {}".format(nbur), self.input_contents
        )

    @property
    def input_file(self):
        return join(self.working_directory, self._INPUT_FILE_FMT.format(self.name))

    @property
    def output_file(self):
        return join(self.working_directory, self._OUTPUT_FILE_FMT.format(self.name))

    def write_inputs(self):
        with open(self.input_file, "w") as f:
            f.write(self.input_contents)
        with open(join(self.working_directory, self._FILENAMES["symmetry"]), "w") as f:
            f.write(self.symmetry_contents)

    def resolve_dependencies(self):
        """ Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        self.adjust_dmain()
        if self.adjust_cutm_nbur:
            self.set_cutm_nbur()
        if not self.spline:
            self.input_contents = re.sub(
                SPLI_REGEX, "", self.input_contents)
        self.write_inputs()

    def result(self):
        return self.output_contents

    def post_process(self):
        files_to_read = {
            name: self._FILENAMES[name]
            for name in ("summary", "shelx", "final_cell", "fdat", "star_plut")
        }

        for name, location in files_to_read.items():
            location = join(self.working_directory, location)
            if exists(location):
                with open(location) as f:
                    setattr(self, name + "_contents", f.read())
                if not self.keep_files:
                    remove(location)

        if exists(self.output_file):
            with open(self.output_file) as f:
                self.output_contents = f.read()
            if not self.keep_files:
                remove(self.output_file)

    def run_checked(self, *args, **kwargs):
        """Run the calculation, may throw exceptions"""
        spacegroup = None
        if self.crystal_info:
            if 'spacegroup' in self.crystal_info.keys():
                spacegroup = self.crystal_info["spacegroup"]
                # DMACRYS sometimes gets the wrong ZVAL so we should calculate it ourselves
                if 'Zp' in self.crystal_info.keys() and not 'ZVAL' in self.kwargs:
                    Zp = self.crystal_info["Zp"]
                    n_symops = n_symops_from_SG(SG = spacegroup, context = 'DMACRYS')
                    self.kwargs['ZVAL'] = Zp * n_symops
        self.resolve_dependencies()
        returncode = 130 
        line_timeout = self.timeout / 30
        t1 = time.time()
        with open(self.input_file) as inp:
            with open(self.output_file, "wb+") as of:
                cmd_list = [self.executable] + [x for x in args]
                exe = Popen(cmd_list, stdout=of, stdin=inp, cwd=self.working_directory)
                checker = DmacrysOutputChecker(
                    self.output_file,
                    exe.pid,
                    period=1.0,
                    content_timeout=line_timeout,
                    kill_timeout=self.timeout,
                    spacegroup=spacegroup,
                    pilot_timeout=self.pilot_timeout
                )
                try:
                    stdout, stderr = exe.communicate(input, timeout=self.timeout)
                except TimeoutExpired:
                    import signal

                    exe.send_signal(signal.SIGKILL)
                    stdout, stderr = exe.communicate()
                    raise TimeoutExpired(
                        exe.args, self.timeout, output=stdout, stderr=stderr
                    )
                returncode = exe.returncode
        t2 = time.time()
        msg = f"Dmacrys exited with return code {returncode} after {t2 - t1:.2f}s"
        self.error = checker.error
        self.post_process()
        if returncode != 0:
            raise ReturnCodeError(msg)
        return returncode

    def run(self, *args, **kwargs):
        try:
            self._run_raw_stdin()
        except ReturnCodeError as e:
            LOG.error("Dmacrys failed: %s", e)
            self.post_process()
            raise e
