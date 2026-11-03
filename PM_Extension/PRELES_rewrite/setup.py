import torch
from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='preles',
    
    ext_modules=[CppExtension('preles',['call_preles.cpp','initruns.cpp', 'preles.cpp', 'gpp.cpp', 'water.cpp'],  
        extra_compile_args=['-g'],                   # Add -g flag for debugging
        include_dirs=cpp_extension.include_paths(),  # Include PyTorch headers
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


################## To RUN ##########
"""
Via Terminal
Activate Environment
Naviate to Folder

python setup.py install

"""



# Extension(
#    name='preles',
#    sources=['call_preles.cpp','initruns.cpp', 'preles.cpp', 'gpp.cpp', 'water.cpp'],
#    include_dirs=cpp_extension.include_paths(),
#    language='c++')