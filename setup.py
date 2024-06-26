from setuptools import setup

setup(
    name="torch-ttnn",
    version="0.1.0",
    description="PyTorch dynamo backend for Tenstorrent TT-NN framework",
    author="yoco",
    author_email="yoco@skymizer.com",
    license="MIT",
    url="https://git.skymizer.com/tenstorrent-metal/pytorch2.0_ttnn",
    packages=["torch_ttnn"],
    install_requires=[],
)
