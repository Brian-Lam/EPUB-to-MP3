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
            chapter_text[chapter_index] += u"{}".format(text) #.extend(t.strip('\n') for t in text) # u"{}".format(text.strip('\n'))
        except IndexError:
            print("Index was {}, object is of length {}".format(
                chapter_index, len(chapter_text)))

    # for i, c in enumerate(chapter_text):
    #     chapter_text[i] = u"".join(c)
    return chapter_text
