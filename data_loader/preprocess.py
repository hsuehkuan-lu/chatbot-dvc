import re
import os
import csv
import codecs

MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]


class ChatbotDataPreprocess:
    """Chatbot dataset."""
    def __init__(self, data_dir, sent_len):
        self.data_dir = data_dir
        self.delimiter = '\t'
        self.sent_len = sent_len
        lines = self._load_lines()
        conversations = self._load_conversations(lines)
        self._dump_file(conversations)

    def _load_lines(self):
        lines = {}
        filename = os.path.join(self.data_dir, 'movie_lines.txt')
        with open(filename, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(' +++$+++ ')
                fields = {}
                for idx, field in enumerate(MOVIE_LINES_FIELDS):
                    fields[field] = values[idx]
                lines[fields['lineID']] = fields
        return lines

    def _load_conversations(self, lines):
        conversations = []
        filename = os.path.join(self.data_dir, 'movie_conversations.txt')
        with open(filename, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(' +++$+++ ')
                # Extract fields
                fields = {}
                for i, field in enumerate(MOVIE_CONVERSATIONS_FIELDS):
                    fields[field] = values[i]
                # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                utterance_id_pattern = re.compile('L[0-9]+')
                line_ids = utterance_id_pattern.findall(fields['utteranceIDs'])
                # Reassemble lines
                fields['lines'] = []
                for line_id in line_ids:
                    fields['lines'].append(lines[line_id])
                conversations.append(fields)
        return conversations

    def _extract_sentence_pairs(self, conversations):
        idx = 0
        qa_pairs = []
        for conversation_id, conversation in enumerate(conversations):
            # Iterate over all the lines of the conversation
            for i in range(len(conversation['lines']) - 1):  # We ignore the last line (no answer for it)
                input_line = conversation['lines'][i]['text'].strip()
                target_line = conversation['lines'][i + 1]['text'].strip()
                # Filter wrong samples (if one of the lists is empty)
                len_input_line = len(input_line.split())
                len_target_line = len(target_line.split())
                if input_line and target_line and \
                        len_input_line < self.sent_len and len_target_line < self.sent_len:
                    qa_pairs.append([idx, conversation_id, input_line, target_line])
                    idx += 1
        return qa_pairs

    def _dump_file(self, conversations):
        filename = 'data/formatted_movie_lines.csv'
        # Unescape the delimiter
        delimiter = str(codecs.decode(self.delimiter, encoding='unicode_escape'))
        headers = ['id', 'conversation_id', 'talk', 'response']
        with open(filename, 'w', encoding='utf-8') as o:
            writer = csv.writer(o, delimiter=delimiter, lineterminator='\n')
            writer.writerow(headers)
            for pair in self._extract_sentence_pairs(conversations):
                writer.writerow(pair)

        self.print_lines(filename)

    @staticmethod
    def print_lines(filename, n=10):
        with open(filename, 'rb') as datafile:
            lines = datafile.readlines()
        for line in lines[:n]:
            print(line)
