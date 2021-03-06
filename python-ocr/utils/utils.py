import glob
import pytesseract
import re


def extract(path='data/*.jpg'):
	pages = glob.glob(path)
	pages.sort()
#	pages = pages[:3]

	text = ''

	for page in pages:
		print('Extracting {}'.format(page))
		image_to_string = pytesseract.image_to_string(page)
		text += image_to_string

	lines = text.split('\n')

	return lines

def build_chapters(lines):
	chapters = {}
	cur_chapter = "Intro"

	for line in lines:
		is_chapter = re.match(r"^(Chapter [0-9]+:)", line)

		if is_chapter:
			cur_chapter = line

		elif cur_chapter in chapters.keys():
			content = "{}\n".format(line)
			chapters[cur_chapter] += content
		else:
			content = "{}\n".format(line)
			chapters[cur_chapter] = content
	return chapters

def convert_chapter_to_spinal(chapter):
	name = re.sub(r"^(Chapter [0-9]+: )", '', chapter)
	if name == chapter:
		raise InvalidChapterException
	name = name.lower().replace(' ', '-')
	return name

class InvalidChapterException(Exception):
	pass

def get_chapter_file(chapter):
	chapter_spinal_case = convert_chapter_to_spinal(chapter)
	return '{}.html'.format(chapter_spinal_case)


def build_html_files(chapters, dest="html/"):
	chapter_keys = list(chapters)
	for index, chapter in enumerate(chapter_keys):
		chapter_file = get_chapter_file(chapter)
		file_name = "{0}{1}".format(dest, chapter_file)
		html_file = open(file_name, 'w')
		prev_link = ''
		next_link = ''
		if index > 0:
			prev_chapter = chapter_keys[index-1]
			prev_chapter_file = get_chapter_file(prev_chapter)
			prev_link = '<p><a href="{}">Pevious</a></p>'.format(prev_chapter_file)

		if (index < len(chapters)-1):
			next_chapter = chapter_keys[index+1]
			next_chapter_file = get_chapter_file(next_chapter)
			next_link = '<p><a href="{}">Next</a></p>'.format(next_chapter_file)
		paragraph = chapters[chapter].replace('\n\n', '<br/><br/>')
		content = """
<html>
    <head>
        <link rel="stylesheet" href="styles.css">
    </head>
    <body>
        <div>
            <h1>{0}</h1>
            <p>{1}</p>
            {2}{3}
        </div>
    </body>
</html>
""".format(chapter, paragraph, prev_link, next_link)

		html_file.write(content)
		html_file.close()




