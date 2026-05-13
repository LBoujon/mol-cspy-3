"""Module to store jinja2 templates for cspy"""
from os.path import dirname, abspath, split
import logging
from jinja2 import Environment, FileSystemLoader, Template

LOG = logging.getLogger(__name__)

CSPY_TEMPLATE_ENV = Environment(loader=FileSystemLoader(dirname(abspath(__file__))))

# Object to store all templates
_ALL_TEMPLATES = {
    "pbs": CSPY_TEMPLATE_ENV.get_template("pbs.template"),
    "slurm": CSPY_TEMPLATE_ENV.get_template("slurm.template"),
    "gaussian_scf": CSPY_TEMPLATE_ENV.get_template("gaussian_scf.template"),
    "gaussian_scf_flex": CSPY_TEMPLATE_ENV.get_template("gaussian_scf_flex.template"),
    "slurm_script_fmcsp": CSPY_TEMPLATE_ENV.get_template("slurm_script_fmcsp.template"),
    "vasic_script_fmcsp": CSPY_TEMPLATE_ENV.get_template("vasic_script_fmcsp.template"),
    "vasopt_script_fmcsp": CSPY_TEMPLATE_ENV.get_template("vasopt_script_fmcsp.template"),
    "vasopt_check_script_fmcsp": CSPY_TEMPLATE_ENV.get_template("vasopt_check_script_fmcsp.template"),
    "test_gaussian_scf": CSPY_TEMPLATE_ENV.get_template("test_gaussian_scf.template"),
    "neighcrys_axis": CSPY_TEMPLATE_ENV.get_template("neighcrys_axis.template"),
    "smult": CSPY_TEMPLATE_ENV.get_template("smult.template"),
    "gdma": CSPY_TEMPLATE_ENV.get_template("gdma.template"),
    "neighcrys_stdin": CSPY_TEMPLATE_ENV.get_template("neighcrysin.template"),
    "pmin": CSPY_TEMPLATE_ENV.get_template("pmin.template"),
    "mulfit": CSPY_TEMPLATE_ENV.get_template("mulfit.template"),
}


def add_template(text=None, filename=None, name="new_template"):
    if filename:
        path, filename = split(filename)
        _ALL_TEMPLATES[name] = Environment(
            loader=FileSystemLoader(path or "./")
        ).get_template(filename)
    elif text:
        _ALL_TEMPLATES[name] = Template(text)
    return _ALL_TEMPLATES[name]


def get_template(name):
    result = _ALL_TEMPLATES.get(name)
    if result is None:
        try:
            return CSPY_TEMPLATE_ENV.get_template(name)
        except Exception as e:
            LOG.error("Could not find template: %s (%s)", name, e)
    return result
