# AgriSatPy - Agricultural Satellite Data Handling in Python

AgriSatPy is a lightweight package to **explore**, **organize** and **process** (satellite) remote sensing data in an easy and intuitive manner.

Developed for **agricultural remote sensing applications** with
**Sentinel-2**, this is still the main thematic focus. However, due to its **modular and object-oriented programming structure**, it allows in principle the **processing of any type of raster data** and can
be **adapted** to **other remote sensing platforms** or **raster data sources** (e.g., Digital Elevation Models, Land Cover Maps, etc.).

## The philosophy:

We believe that researchers and analysts should **deal as little as possible with file handling and backend engineering**. In addition, the underlying source code should be **open source** and non-proprietary.
However, in the field of optical remote sensing it is still difficult to analyse satellite data without being confronted with these tasks or resorting to proprietary solutions.
Therefore, we have developed AgriSatPy in such a way that a large part of these tasks is taken away from the user and provided in the form of self-explanatory attributes and methods on a high semantic level.

AgriSatPy can be used for **rapid prototyping**. At the same time, AgriSatPy also offers support for the **operational use of satellite data** - for example by maintaining a metadatabase and providing API-level
 functions. In contrast to related projects such as the [OpenDataCube](https://www.opendatacube.org/) Initiative, AgriSatPy can be used without the need for complex installations - many of the basic functions
 for interacting with raster data follow the " **plug-and-play** " principle.

AgriSatPy can be used by users with little coding experience as well as by experienced developers. This is made possible by the object-oriented and modular structuring of the code, as well as the detailed
documentation.

AgriSatPy provides **interfaces** to widely used Python libraries such as [xarray](https://xarray.pydata.org/en/stable/#) and [geopandas](https://geopandas.org/en/stable/) and uses [rasterio](https://rasterio.readthedocs.io/en/latest/) and [numpy](https://numpy.org/) as backend.

## Why AgriSatPy

coming soon

## Capabilities

coming soon

## Code Documentation

We use [sphinx](https://www.sphinx-doc.org/en/master/) and the [autodoc-extension](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) to generate a Read-The-Docs like documentation of our modules, classes and functions.

With ```sphinx-apidoc -o ./source ../src``` run in the ```./docs``` directory you can generate the required .rst files if not yet available.

## Contributing
Yes please :)

Before you start, please:
- familiarize yourself working with `git` if you haven't used it before. Always work in **your own fork** and **never in the master branch**.
- use meaningfull commit messages and try to commit your changes with a certain granularity (e.g, module-wise).
- All changes must be submitted using `merge requests`. The merge request should also have a small message text where you explain what this merge will change.
- make sure to follow the [coding style conventions](./CODE_STYLE.md).
- provide test (pytest) cases whenever possible.

## Versioning Rules

We use [setuptools-scm](https://pypi.org/project/setuptools-scm/) with a post-release scheme for dynamic versioning based on git commit IDs and git tags.

A **release** is a stable version of the package and should be used for operational purposes. Each release should have a version. The versioning system used is `vMajor.Minor (e.g., v1.1)`.
In addition to releases we have `development versions`. This also receive version numbers based on a `post-release` versioning logic. This consists of the latest stable version (i.e., the base for the development), plus the number of the commit since the last release. An example for such a development version is:

```bash
v1.1.post44+gaaa8b16
```

In this example, `v1.1` indicates that v1.1 was the last stable release on which the current development version is based. `post` means that the current version was created *after* this release. `44` is the number of commits since the last release and `gaaa8b16` a shortened version of commit UID.

To create a new release, commit and push all your changes into your development branch and merge them into the main project's master.

Next, checkout the master branch and pull the merged changes locally. Then add a git tag (by increasing the version number):

```bash
git checkout master
git pull upstream master
git tag -a v<major>.<minor> -m "v<major>.<minor>"
git push upstream master
```
Then publish the package to the PyPI package index by running `publish_pypi.sh` or `publish_pypi.bat` depending on your OS.

**NOTE**: git tags won't be fetched by simply pulling from the upstream's master branch. They must be fetched using

```bash
git fetch upstream
```

You can verify that you have received the tag by typing `git tag -l` to get a list of all available git tags.

Then checkout your development branch again.

**IMPORTANT**:

- New releases should always be discussed with the other developers.
- Only the package administrator should be able to publish new releases.
- Always add the most important changes to changelog so that it will be possible to track the development between the releases

