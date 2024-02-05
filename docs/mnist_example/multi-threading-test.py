#!/usr/bin/env python3
# encoding=utf-8

# testing for multi-threading with onnx-mlir compilation

import datetime
import os
import threading


def execCmd(cmd):
    try:
        print("command " + cmd + " starts at " + str(datetime.datetime.now()))
        os.system(cmd)
        print("command " + cmd + " is finished at " + str(datetime.datetime.now()))
    except:
        print("command " + cmd + " meets errors")


if __name__ == "__main__":
    # define 2 different commands
    cmds = [
        "onnx-mlir -O3 mnist.onnx -o mnist03",
        "onnx-mlir -O1 mnist.onnx -o mnist01",
    ]

    threads = []

    print("program starts at " + str(datetime.datetime.now()))

    # run the commands
    for cmd in cmds:
        th = threading.Thread(target=execCmd, args=(cmd,))
        th.start()
        threads.append(th)

    # wait for all the commands finish
    for th in threads:
        th.join()

    print("program is finished at " + str(datetime.datetime.now()))
