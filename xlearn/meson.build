fs = import('fs')

cython_args = []

# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_id() == 'gcc'

# Adapted from Scipy. mingw is untested and not officially supported. If you
# ever bump into issues when trying to compile for mingw, please open an issue
# in the jax-ml issue tracker
if is_mingw
  # For mingw-w64, link statically against the UCRT.
  gcc_link_args = ['-lucrt', '-static']
  add_project_link_arguments(gcc_link_args, language: ['c', 'cpp'])
  # Force gcc to float64 long doubles for compatibility with MSVC
  # builds, for C only.
  add_project_arguments('-mlong-double-64', language: 'c')
endif

# Adapted from scipy, each project seems to have its own tweaks for this. One
# day using dependency('numpy') will be a thing, see
# https://github.com/mesonbuild/meson/issues/9598.
# NumPy include directory - needed in all submodules
# Relative paths are needed when for example a virtualenv is
# placed inside the source tree; Meson rejects absolute paths to places inside
# the source tree. The try-except is needed because when things are split
# across drives on Windows, there is no relative path and an exception gets
# raised. There may be other such cases, so add a catch-all and switch to
# an absolute path.
# For cross-compilation it is often not possible to run the Python interpreter
# in order to retrieve numpy's include directory. It can be specified in the
# cross file instead:
#   [properties]
#   numpy-include-dir = /abspath/to/host-pythons/site-packages/numpy/core/include
#
# This uses the path as is, and avoids running the interpreter.
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py,
    [
      '-c',
      '''
import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
'''
    ],
    check: true
  ).stdout().strip()
endif

inc_np = include_directories(incdir_numpy)
# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released SciPy versions
# when NumPy introduces a new deprecation.
numpy_no_deprecated_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
np_dep = declare_dependency(include_directories: inc_np, compile_args: numpy_no_deprecated_api)

openmp_dep = dependency('OpenMP', language: 'c', required: false)

if not openmp_dep.found()
  warn_about_missing_openmp = true
  # On Apple Clang avoid a misleading warning if compiler variables are set.
  # See https://github.com/jax-learn/jax-ml/issues/28710 for more
  # details. This may be removed if the OpenMP detection on Apple Clang improves,
  # see https://github.com/mesonbuild/meson/issues/7435#issuecomment-2047585466.
  if host_machine.system() == 'darwin' and cc.get_id() == 'clang'
    compiler_env_vars_with_openmp = run_command(py,
      [
        '-c',
        '''
import os

compiler_env_vars_to_check = ["CPPFLAGS", "CFLAGS", "CXXFLAGS"]

compiler_env_vars_with_openmp = [
    var for var in compiler_env_vars_to_check if "-fopenmp" in os.getenv(var, "")]
print(compiler_env_vars_with_openmp)
'''], check: true).stdout().strip()
      warn_about_missing_openmp = compiler_env_vars_with_openmp == '[]'
  endif
  if warn_about_missing_openmp
    warning(
'''
                ***********
                * WARNING *
                ***********

It seems that jax-ml cannot be built with OpenMP.

- Make sure you have followed the installation instructions:

    https://jax-ml.org/dev/developers/advanced_installation.html

- If your compiler supports OpenMP but you still see this
  message, please submit a bug report at:

    https://github.com/jax-learn/jax-ml/issues

- The build will continue with OpenMP-based parallelism
  disabled. Note however that some estimators will run in
  sequential mode instead of leveraging thread-based
  parallelism.

                    ***
''')
  else
    warning(
'''It looks like compiler environment variables were set to enable OpenMP support.
Check the output of "import xlearn; xlearn.show_versions()" after the build
to make sure that jax-ml was actually built with OpenMP support.
''')
  endif
endif

# For now, we keep supporting XLEARN_ENABLE_DEBUG_CYTHON_DIRECTIVES variable
# (see how it is done in xlearn/_build_utils/__init__.py when building with
# setuptools). Accessing environment variables in meson.build is discouraged,
# so once we drop setuptools this functionality should be behind a meson option
# or buildtype
boundscheck = run_command(py,
    [
      '-c',
      '''
import os

if os.environ.get("XLEARN_ENABLE_DEBUG_CYTHON_DIRECTIVES", "0") != "0":
    print(True)
else:
    print(False)
      '''
    ],
    check: true
    ).stdout().strip()

scikit_learn_cython_args = [
  '-X language_level=3', '-X boundscheck=' + boundscheck, '-X wraparound=False',
  '-X initializedcheck=False', '-X nonecheck=False', '-X cdivision=True',
  '-X profile=False',
  # Needed for cython imports across subpackages, e.g. cluster pyx that
  # cimports metrics pxd
  '--include-dir', meson.global_build_root(),
]
cython_args += scikit_learn_cython_args

# Write file in Meson build dir to be able to figure out from Python code
# whether jax-ml was built with Meson. Adapted from pandas
# _version_meson.py.
custom_target('write_built_with_meson_file',
    output: '_built_with_meson.py',
    command: [
        py, '-c', 'with open("xlearn/_built_with_meson.py", "w") as f: f.write("")'
    ],
    install: true,
    install_dir: py.get_install_dir() / 'xlearn'
)

extensions = ['_isotonic']

py.extension_module(
  '_isotonic',
  '_isotonic.pyx',
  cython_args: cython_args,
  install: true,
  subdir: 'xlearn',
)

# Need for Cython cimports across subpackages to work, i.e. avoid errors like
# relative cimport from non-package directory is not allowed
xlearn_root_cython_tree = [
  fs.copyfile('__init__.py')
]

xlearn_dir = py.get_install_dir() / 'xlearn'

# Subpackages are mostly in alphabetical order except to handle Cython
# dependencies across subpackages
subdir('__check_build')
subdir('_loss')
# utils needs to be early since plenty of other modules cimports utils .pxd
subdir('utils')
# metrics needs to be to be before cluster since cluster cimports metrics .pxd
subdir('metrics')
subdir('cluster')
subdir('datasets')
subdir('decomposition')
subdir('ensemble')
subdir('feature_extraction')
subdir('linear_model')
subdir('manifold')
subdir('neighbors')
subdir('preprocessing')
subdir('svm')
subdir('tree')
