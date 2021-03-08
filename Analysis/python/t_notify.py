#!/usr/bin/env python
# Send notification via telegram

import os
import sys

homeDir = os.path.expanduser('~')
tokenFileName = os.path.join(homeDir, 'private', 'bot.token')
chatIdFileName = os.path.join(homeDir, 'private', 'bot.chat_id')

def ReadString(fileName):
    f = open(fileName, 'r')
    lines = [ s.strip() for s in f.readlines() ]
    if len(lines) < 1:
        raise RuntimeError('The file "{}" is empty.'.format(fileName))
    return lines[0]

token = ReadString(tokenFileName)
chatId = int(ReadString(chatIdFileName))

import telepot
bot = telepot.Bot(token)

def Notify(message):
    try:
        bot.sendMessage(chatId, message)
    except:
        print("ERROR: unable to send notification.", sys.exc_info()[0])

if __name__ == "__main__":
    message = 'Notification'
    if len(sys.argv) >= 2:
        message = sys.argv[1]
    Notify(message)
