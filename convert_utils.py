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


"""
Returns array of chapter text from an epub files full text.
"""


def get_all_chapters(full_text):
    """Parse the full xml encoded text of an epub file and return an array of strings
    corresponding to the chapters of the epub

    Args:
        full_text (str): The xml encoded contents of an entire .epub file

    Returns:
        list(str): A list of string objects, each of which corresponds to 
        a chapter of the .epub file, i.e. ["chapter 1 text", "chapter 2 text", ...]
    """    
    soup = BeautifulSoup(full_text, 'html.parser')
    all_text = soup.find_all(text=True)
    visible_texts = filter(tag_visible, all_text)

    chapters = soup.find_all('h2')  # All chapter headings
    chapters = {next(c.children): i+1 for i, c in enumerate(chapters)}
    num_chapters = max(chapters.values())

    chapter_index = -1  # The next chapter heading
    chapter_text = ['' for _ in range(num_chapters+1)]
    for text in visible_texts:
        if text in chapters:
            chapter_index = chapters[text]
            print("Now at chapter {}".format(chapter_index))
        if chapter_index < 0:
            continue
        try:
            chapter_text[chapter_index] += u"{}".format(text)
        except IndexError:
            print("Index was {}, object is of length {}".format(
                chapter_index, len(chapter_text)))
    return chapter_text
