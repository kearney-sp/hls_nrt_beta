import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='src',
    version='0.0.1',
    author='Sean Kearney',
    author_email='sean.patrick@hotmail.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/kearney-sp/hls_nrt_beta/src',
    project_urls = {
        "Bug Tracker": "https://github.com/kearney-sp/hls_nrt_beta/issues"
    },
    license='MIT',
    packages=['toolbox'],
    install_requires=['requests'],
)