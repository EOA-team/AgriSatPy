agrisatpy.metadata package
==========================

The agrisatpy.metadata package allows for creating, maintaining and querying a satellite metadata database.
The metadata DB is the key requirement for all operational functions.

To make this agrisatpy.metadata package work a running PostgreSQL DBMS instance is required. Additionally, the
spatial PostGIS extension must be enabled.

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   agrisatpy.metadata.database
   agrisatpy.metadata.sentinel2

Module contents
---------------

.. automodule:: agrisatpy.metadata
   :members:
   :undoc-members:
   :show-inheritance:
