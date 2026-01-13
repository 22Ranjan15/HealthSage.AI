import os
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    
    if not os.path.exists(file_path):
        return []

    with open(file_path) as file_obj:
        lines = file_obj.readlines()
        lines = [req.replace("\n", "") for req in lines]

        for req in lines:
            # 1. Ignore editable install triggers
            if req == HYPEN_E_DOT:
                continue
            
            # 2. Ignore comments and empty lines
            if req.startswith('#') or not req.strip():
                continue

            # 3. ADVANCED: Handle recursive references (-r backend/requirements.txt)
            if req.startswith('-r'):
                included_file = req.split(' ')[1]
                requirements.extend(get_requirements(included_file))
            else:
                requirements.append(req)

    return requirements

setup(
    name="HealthSage.AI",
    version="0.0.1",
    author="Ranjan",
    author_email="ranjandasbd22@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements-dev.txt')
)