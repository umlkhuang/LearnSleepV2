__author__ = 'ke'

from glob import glob
from database import DatabaseHelper

if __name__ == "__main__":
    dbList = glob('./data/*.db')
    for dbName in dbList:
        dbHelper = DatabaseHelper(dbName)
        dbHelper.addColumns()

