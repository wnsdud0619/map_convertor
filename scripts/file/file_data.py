#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from abc import (
    ABCMeta,
    abstractmethod,
)
from functools import (
    reduce,
)
import sqlite3

from six import add_metaclass

from scripts.core.core_data import (
    Singleton,
)
from scripts.functions.file_functions import (
    check_file,
    create_directory,
)
from scripts.functions.print_functions import (
    Process_Counter,
    log_print,
    warning_print,
    error_print,
)


@add_metaclass(ABCMeta)
class Core(Singleton):
    
    @classmethod
    def _open(cls, file_path):
        
        if not check_file(file_path):
            dir_path = reduce(lambda x, y : "{0}/{1}".format(x, y), file_path.split("/")[:-1])
            create_directory(dir_path)

        connector = sqlite3.connect(file_path, isolation_level=None)
        
        # - 속도 향상
        connector.execute("pragma synchronous=OFF")
        connector.execute("begin transaction")
        
        cursor = connector.cursor()

        cls().connector = connector
        cls().cursor = cursor

    @classmethod
    def _close(cls):
        cls().connector.commit()
        cls().cursor.close()

    # --------------------------------------------------

    @classmethod
    def _execute(cls, command):
        cls().cursor.execute(command)  

    @classmethod
    def _executemany(cls, command, tuples):
        cls().cursor.executemany(command, tuples)

    @classmethod
    def _create_table(cls, table_name, column_list):
        cls._execute("create table if not exists {0} ({1})".format(table_name, column_list))

    @classmethod
    def _check_table(cls, table_name):
        """
        테이블 유무 확인
        """
        cls._execute("select name from sqlite_master where type='table' and name='{0}'".format(table_name))
        if cls().cursor.fetchone() == None:
            # log_print("Test 1 = {0}".format(table_name))
            return False
        # log_print("Test 2 = {0}".format(table_name))
        return True

    @classmethod
    def _check_columns(cls, table_name, columns):
        """
        테이블 항목 확인 
        """

        cls._execute("select * from {0}".format(table_name))
        _columns = [d[0] for d in cls().cursor.description]

        if any([column not in _columns for column in columns]):
            return False
        return True

    @classmethod
    def _delete_all(cls, table_name):
        if cls._check_table(table_name):
            cls._execute("delete from {0}".format(table_name))

    @classmethod
    def _drop(cls, table_name):
        cls._execute("drop table {0}".format(table_name))

    @classmethod
    def _load_records(cls, table_name):
        cls._execute("select * from {0}".format(table_name))
        records = cls().cursor.fetchall()
        return records


@add_metaclass(ABCMeta)
class Saver():

    @abstractmethod
    def _save_data(self, file_type, file_path):
        pass

    # --------------------------------------------------

    @classmethod
    def save(cls, file_path):

        if not cls()._save_data(file_path):
            return True

        warning_print("Failed to save file")
        return False


@add_metaclass(ABCMeta)
class Loader():

    @abstractmethod
    def _load_data(self, file_type, file_path):
        pass

    # --------------------------------------------------

    @classmethod
    def load(cls, file_path):

        if not cls()._load_data(file_path):
            return True

        warning_print("Failed to load file")
        return False
