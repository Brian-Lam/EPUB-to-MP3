import time
from bs4 import BeautifulSoup

""" 
A helper function to return the current millisecond timestamp.
"""
def current_milli_time():
    return int(round(time.time() * 1000))

""" 
A helper function to omit elements from certain types of tags in 
an epub file.
"""
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    return True

"""
Returns just the text from an epub file.
"""
def chapter_to_text(chapter_xml_contents):
    output = ''
    soup = BeautifulSoup(chapter_xml_contents, 'html.parser')
    all_text = soup.find_all(text=True)
    visible_texts = filter(tag_visible, all_text)  
    return u" ".join(t.strip() for t in visible_texts)
