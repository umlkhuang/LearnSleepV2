from util import Counter 
from constants import SLEEP_COLLECTOR_NAME 

class SensingData(object):
    """
    Stores all information of an entry of sensing data. 
    """
    
    def __init__(self, createTime, trackDate, movement, illuminanceMax, illuminanceMin, illuminanceAvg, illuminanceStd,
                 decibelMax, decibelMin, decibelAvg, decibelStd, isCharging, powerLevel, proximity, ssid, appUsage):
        self.createTime = createTime
        self.trackDate = trackDate
        self.month = self.createTime.month 
        self.day = self.createTime.day
        self.dayOfWeek = self.createTime.weekday() 
        self.hour = self.createTime.hour 
        self.movement = movement
        self.illuminanceMax = illuminanceMax 
        self.illuminanceMin = illuminanceMin
        self.illuminanceAvg = illuminanceAvg
        self.illuminanceStd = illuminanceStd
        self.decibelMax = decibelMax 
        self.decibelMin = decibelMin
        self.decibelAvg = decibelAvg
        self.decibelStd = decibelStd
        self.isCharging = isCharging
        self.powerLevel = powerLevel
        self.proximity = proximity 
        self.ssid = ssid
        self.appUsage = appUsage 
        
    def __repr__(self):
        dateStr = self.createTime.strftime("%Y-%m-%d %H:%M:%S")
        return "%s | %s | %d | %f | %f | %f | %d | %d | %d | %d | %f | %f | %s | %s" % (dateStr, self.trackDate, self.movement, self.illuminanceMax, self.illuminanceMin, self.illuminanceAvg,
                    self.decibelMax, self.decibelMin, self.decibelAvg, self.isCharging, self.powerLevel, self.proximity, self.ssid, self.appUsage)


class SleepLog(object):
    """
    Stores all information of an entry of sleep log data.
    """
    
    def __init__(self, createTime, trackDate, sleepTime, wakeupTime, quality, finished):
        self.createTime = createTime
        self.trackDate = trackDate
        self.sleepTime = sleepTime
        self.wakeupTime = wakeupTime
        self.quality = quality
        self.finished = finished
        
    def __repr__(self):
        createStr = self.createTime.strftime("%Y-%m-%d %H:%M:%S")
        sleepStr = self.sleepTime.strftime("%Y-%m-%d %H:%M:%S")
        wakeupStr = self.wakeupTime.strftime("%Y-%m-%d %H:%M:%S")
        return "%s | %s | %s | %s | %d | %d" % (createStr, self.trackDate, sleepStr, wakeupStr, self.quality, self.finished)


class SysEvent(object):
    """
    Stores all information of an entry of system event data.
    """
    
    def __init__(self, createTime, trackDate, eventType):
        self.createTime = createTime
        self.trackDate = trackDate
        self.eventType = eventType
        self.events = ["Power_On", "Power_Off", "Screen_On", "Screen_Off", "Power_Connected", "Power_Disconnected", "Logged_Sleep", "Logged_Wakeup"]
        
    def __repr__(self):
        createStr = self.createTime.strftime("%Y-%m-%d %H:%M:%S")
        return "%s | %s | %s" % (createStr, self.trackDate, self.events[self.eventType - 1])
    

class CombinedData(object):
    """
    The class that stores the combined data. 
    """
    
    def __init__(self, sensingData):
        self.createTime = sensingData.createTime 
        self.month = sensingData.month 
        self.day = sensingData.day 
        self.dayOfWeek = sensingData.dayOfWeek 
        self.hour = sensingData.hour 
        self.movement = sensingData.movement 
        self.illuminanceMax = sensingData.illuminanceMax
        self.illuminanceMin = sensingData.illuminanceMin
        self.illuminanceAvg = sensingData.illuminanceAvg
        self.illuminanceStd = sensingData.illuminanceStd
        self.decibelMax = sensingData.decibelMax 
        self.decibelMin = sensingData.decibelMin
        self.decibelAvg = sensingData.decibelAvg
        self.decibelStd = sensingData.decibelStd
        self.isCharging = sensingData.isCharging
        self.powerLevel = sensingData.powerLevel 
        self.proximity = sensingData.proximity 
        self.screenOnSeconds, self.appUsage = self.processAppUsage(sensingData.appUsage) 
        self.wifiCounter = self.processWifi(sensingData.ssid) 
        self.isSleep = None  
        
    def processAppUsage(self, appUsage):
        """
        Parse the appUsage raw record and return a counter of app usage.
        The Sleep Collector and Launcher application is excluded. 
        """ 
        usageCounter = Counter() 
        screenOnSeconds = 0 
        if appUsage != "":
            usageList = appUsage.split(",")
            for oneUsage in usageList:
                tmp = oneUsage.split(":")
                if len(tmp) != 3:
                    continue
                else: 
                    screenOnSeconds += int(tmp[2])
                    if tmp[0] == SLEEP_COLLECTOR_NAME:
                        continue 
                    else: 
                        usageCounter[tmp[0]] += int(tmp[2]) 
        return screenOnSeconds, usageCounter 
    
    def processWifi(self, ssid):
        """
        Parse the WiFi signature raw data and return a counter of WiFi SSID.
        """
        wifiCounter = Counter()
        if ssid != "":
            wifiList = ssid.split(", ")
            for oneWifi in wifiList: 
                tmp = oneWifi.split("#") 
                if len(tmp) == 2:
                    wifiCounter[tmp[0]] = int(tmp[1]) 
        return wifiCounter 
    
    def setSleepStatus(self, isSleep):
        """
        Set the sleep status, input is integer, 1 for sleep and 0 for awake 
        """
        self.isSleep = isSleep 
        
    def getSleepStatus(self):
        return self.isSleep 
    
    


    