import re
import math

def extract_answer (raw):
   if len(raw) == 0:
      return ""
   return raw[0]
def extract_sample(raw):
   if len(raw.split("assistant")) > 2:
      return "".join(raw.split("assistant")[2:]).strip()
   else:
      return raw