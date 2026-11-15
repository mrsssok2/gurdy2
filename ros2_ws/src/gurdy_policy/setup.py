import os
from glob import glob

from setuptools import setup

package_name = "gurdy_policy"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="S. Mahadev",
    description="RL policy node (ONNX) for Gurdy",
    license="MIT",
    entry_points={"console_scripts": ["policy_node = gurdy_policy.policy_node:main"]},
)
