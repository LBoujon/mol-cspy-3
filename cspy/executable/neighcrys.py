from .executable import AbstractExecutable, ReturnCodeError
import logging
from os import remove, listdir, mkdir
from os.path import basename, exists, join, splitext
from cspy.templates import get_template
from cspy.executable.locations import NEIGHCRYS_EXEC
import copy


LOG = logging.getLogger("neighcrys")


class Neighcrys(AbstractExecutable):
    _template = get_template("neighcrys_stdin")
    _INPUT_FILE_FMT = "{}.neighcrys_stdin"
    _OUTPUT_FILE_FMT = "{}.neighcrys_stdout"
    _DMACRYS_FILE_FMT = "{}.dmain"
    _DMACRYS_FILE_FMT_OLD = "{}.res.dmain"
    _executable_location = NEIGHCRYS_EXEC

    def __init__(self, res_filename, name=None, working_directory=".", keep_files=False, **kwargs):
        from cspy.configuration import CONFIG

        self.res_filename = res_filename
        self.args = copy.deepcopy(CONFIG.get("neighcrys"))
        self.args.update(**kwargs)
        self.name = name
        self.output_contents = None
        self.dmain_contents = None
        self.symmetry_contents = None
        self.working_directory = working_directory
        self.res_basename = basename(splitext(self.res_filename)[0])
        self.keep_files = keep_files
        if not self.name:
            self.name = self.res_basename

    @property
    def symmetry_filename(self):
        return join(self.working_directory, "fort.20")

    @property
    def fort21(self):
        return join(self.working_directory, "fort.21")

    @property
    def fort22(self):
        return join(self.working_directory, "fort.22")

    @property
    def input_file(self):
        return join(self.working_directory, self._INPUT_FILE_FMT.format(self.name))

    @property
    def dmacrys_input_file(self):
        return join(
            self.working_directory, self._DMACRYS_FILE_FMT.format(self.res_basename)
        )

    @property
    def old_dmacrys_input_file(self):
        return join(
            self.working_directory, self._DMACRYS_FILE_FMT_OLD.format(self.res_basename)
        )

    @property
    def output_file(self):
        return join(self.working_directory, self._OUTPUT_FILE_FMT.format(self.name))

    @property
    def cleanup(self):
        filenames = [
            self.symmetry_filename,
            self.fort21,
            self.fort22,
            self.input_file,
            self.output_file,
            self.output_file,
            self.dmacrys_input_file,
        ]
        for f in filenames:
            yield f

    @property
    def input_contents(self):
        return self._template.render(res_filename=self.res_filename, **self.args)

    def write_inputs(self):
        input_str = self._template.render(res_filename=self.res_filename, **self.args)
        with open(self.input_file, "w") as f:
            f.write(input_str)

    def resolve_dependencies(self):
        """ Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        self.write_inputs()

    def result(self):
        return self.dmain_contents

    def post_process(self):
        if exists(self.output_file):
            with open(self.output_file) as f:
                self.output_contents = f.read()

        if exists(self.fort21):
            with open(self.fort21) as f:
                self.fort21_contents = f.read()

        if exists(self.dmacrys_input_file):
            with open(self.dmacrys_input_file) as f:
                self.dmain_contents = f.read()
        elif exists(self.old_dmacrys_input_file):
            with open(self.old_dmacrys_input_file) as f:
                self.dmain_contents = f.read()
        else:
            LOG.error(
                "Could not read dmain_contents: tried (%s, %s)",
                self.dmacrys_input_file,
                self.old_dmacrys_input_file,
            )
            LOG.error(
                "Files in %s\n%s",
                self.working_directory,
                ",\n".join(listdir(self.working_directory)),
            )
            LOG.error("fort.21 contents:\n%s", self.fort21_contents)

        if exists(self.symmetry_filename):
            with open(self.symmetry_filename) as f:
                self.symmetry_contents = f.read()

        if self.keep_files:
            mkdir(self.working_directory + '/NEIGHCRYS')
            from shutil import move

        for fname in self.cleanup:
            if exists(fname):
                if self.keep_files:
                    move(fname, self.working_directory + '/NEIGHCRYS/' + fname.split('/')[-1])
                else:
                    remove(fname)

    def run(self, *args, **kwargs):
        try:
            self._run_raw_stdin()
        except ReturnCodeError as e:
            LOG.error("Neighcrys failed: %s", e)
            self.post_process()
            LOG.error("output: %s", self.output_contents)
            raise e
