'''
Created on Jul 12, 2021

@author: Lukas Graf (D-USYS; ETHZ)
'''

from sqlalchemy.orm import declarative_base


Base = declarative_base()

class MetaDB(Base):

    __tablename__ = 'sentinel2_metadata'

    # TODO
