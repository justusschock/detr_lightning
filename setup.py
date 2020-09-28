import os

from setuptools import setup, find_packages

def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file, encoding="utf8") as f:
        content = f.read()
    return content


requirements = resolve_requirements(
    os.path.join(os.path.dirname(__file__), "requirements", 'install.txt'))

setup(
    name='detr',
    version='0.0.1',
    packages=find_packages(),
    test_suite="unittest",
    long_description_content_type='text/markdown',
    install_requires=requirements,
    tests_require=["coverage"],
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=False,
)