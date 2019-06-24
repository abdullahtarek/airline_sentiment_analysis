import urllib.request
from zipfile import ZipFile

link_address = "http://nlp.stanford.edu/data/glove.6B.zip"
local_filename = "glove.6B.zip"
urllib.request.urlretrieve( link_address, local_filename)  

with ZipFile(local_filename, 'r') as zipObj:
    zipObj.extractall()