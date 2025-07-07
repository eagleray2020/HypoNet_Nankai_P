from setuptools import setup, find_packages

setup(
    name='HypoNet_Nankai',
    version='0.1.0',
    description='Hypocenter determination with HypoNet Nankai',
    author='Ryoichiro Agata',
    author_email='agatar@jamstec.go.jp',
    packages=find_packages(),
    install_requires=[
        'numpy<2',
        'torch>=2.0.0',
        'matplotlib',
        'scipy',
        'netCDF4',
        'pymap3d',
    ],
    include_package_data=True,
    package_data={
        'HypoNet_Nankai': [
            'input/*',
            'input_s/*',
            'input_p/*',
            'input_p/**/*',
            'input_p/**/**/*',
        ],
    },
    entry_points={
        'console_scripts': [
            'hyponetn_run = HypoNet_Nankai.main:main',
        ],
    },
    python_requires='>=3.7',
)
