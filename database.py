import sqlite3
import sys 
import os

from instance import SensingData, SleepLog, SysEvent

class DatabaseHelper():
    """
    The database class that used to access a sqlite3 database file. It contains all
    necessary functions that used in the project. 
    """
    
    def __init__(self, fullDBPath):
        self.fullDBPath = fullDBPath
        self.con = None
        
        if not os.path.exists(self.fullDBPath):
            print "The path of database file is not correct, please double check the file path."
            sys.exit(1) 
        
        try:
            # http://stackoverflow.com/questions/1829872/read-datetime-back-from-sqlite-as-a-datetime-in-python 
            self.con = sqlite3.connect(self.fullDBPath, detect_types = sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        except sqlite3.Error, e:
            print "Connect to database %s error: %s" % (self.fullDBPath, e.args[0])
            sys.exit(1)
    
    def __del__(self):
        if self.con != None:
            self.con.close()
    
    def getSensingData(self):
        data = []
        sqlstr = "SELECT createTime as \"[timestamp]\", trackDate, movement, illuminanceMax, illuminanceMin, illuminanceAvg, illuminanceStd, \
                    decibelMax, decibelMin, decibelAvg, decibelStd, isCharging, powerLevel, proximity, ssid, appUsage \
                    FROM sensingdata" 
        try:
            cur = self.con.cursor()
            cur.execute(sqlstr)
            rows = cur.fetchall()
            
            for row in rows:
                createTime = row[0]
                trackDate = row[1]
                movement = int(row[2]) 
                illuminanceMax = float(row[3])
                illuminanceMin = float(row[4])
                illuminanceAvg = float(row[5])
                illuminanceStd = float(row[6])
                decibelMax = int(row[7])
                decibelMin = int(row[8])
                decibelAvg = int(row[9])
                decibelStd = int(row[10])
                isCharging = int(row[11])
                powerLevel = float(row[12])
                proximity = float(row[13])
                ssid = row[14]
                appUsage = row[15]
                oneRecord = SensingData(createTime, trackDate, movement, illuminanceMax, illuminanceMin, illuminanceAvg, illuminanceStd,
                         decibelMax, decibelMin, decibelAvg, decibelStd, isCharging, powerLevel, proximity, ssid, appUsage)
                data.append(oneRecord) 
        except sqlite3.Error, e:
            print "Query sensingdata table error: %s" % (e.args[0])
            sys.exit(1)
        finally:
            cur.close()
        return data    
    
    def getSleepLog(self):
        """
        Pull all sleep log from database to a list. We need to sort the record by sleep time
        since we will use this time stamp to combine the raw data. 
        """
        data = []
        sqlstr = "SELECT createTime as \"[timestamp]\", trackDate, sleepTime as \"[timestamp]\", \
                    wakeupTime as \"[timestamp]\", quality, finished FROM sleeplogger \
                    order by date(sleepTime) ASC "
        
        try:
            cur = self.con.cursor()
            cur.execute(sqlstr)
            rows = cur.fetchall()
            
            for row in rows:
                createTime = row[0]
                trackDate = row[1]
                sleepTime = row[2]
                wakeupTime = row[3]
                quality = int(row[4])
                finished = int(row[5])
                oneRecord = SleepLog(createTime, trackDate, sleepTime, wakeupTime, quality, finished)
                data.append(oneRecord)
            return data 
        except sqlite3.Error, e:
            print "Query sleeplogger table error: %s" % (e.args[0])
            sys.exit(1)
        finally:
            cur.close()
    
    def getSystemEvents(self):
        data = []
        sqlstr = "SELECT createTime as \"[timestamp]\", trackDate, eventType FROM sysevents"
        
        try:
            cur = self.con.cursor()
            cur.execute(sqlstr)
            rows = cur.fetchall()
            
            for row in rows:
                createTime = row[0]
                trackDate = row[1]
                eventType = int(row[2])
                oneRecord = SysEvent(createTime, trackDate, eventType)
                data.append(oneRecord)
            return data
        except sqlite3.Error, e:
            print "Query sysevents table error: %s" % (e.args[0])
            sys.exit(1)
        finally:
            cur.close()

    def addColumns(self):
        sqlstr1 = "ALTER TABLE sensingdata ADD COLUMN illuminanceStd FLOAT DEFAULT 0"
        sqlstr2 = "ALTER TABLE sensingdata ADD COLUMN decibelStd FLOAT DEFAULT 0"

        try:
            cur = self.con.cursor()
            cur.execute(sqlstr1)
            cur.execute(sqlstr2)
        except sqlite3.Error, e:
            print "Update sensingdata table error: %s" % (e.args[0])
            sys.exit(1)
        finally:
            cur.close()
    


if __name__ == "__main__":
    test = DatabaseHelper("./data/18dcdfbc751064e9251fa718a9319fe6.db")
    data1 = test.getSensingData()
    data2 = test.getSleepLog()
    data3 = test.getSystemEvents()
    print data1[0]
    print "\n\n" 
    print data2[0] 
    print "\n\n"
    print data3[0]
    
    
    