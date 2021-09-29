import re
import tokenize

from io import BytesIO
from sacrebleu import TOKENIZERS
# tokenize_v14_international

import tokenizer.java_tokenizer as javalang_tok

PYTHON_TOKEN2CHAR = {
    'STOKEN0': '#',
    'STOKEN1': "\\n",
    'STOKEN2': '"""',
    'STOKEN3': "'''"
}

PYTHON_CHAR2TOKEN = {
    '#': ' STOKEN0 ',
    "\\n": ' STOKEN1 ',
    '"""': ' STOKEN2 ',
    "'''": ' STOKEN3 '
}

JAVA_TOKEN2CHAR = {
    'STOKEN0': "//",
    'STOKEN1': "/*",
    'STOKEN2': "*/",
    'STOKEN3': "/**",
    'STOKEN4': "**/",
    'STOKEN5': '"""',
    'STOKEN6': '\\n'
}
JAVA_CHAR2TOKEN = {
    "//": ' STOKEN0 ',
    "/*": ' STOKEN1 ',
    "*/": ' STOKEN2 ',
    "/**": ' STOKEN3 ',
    "**/": ' STOKEN4 ',
    '"""': ' STOKEN5 ',
    '\\n': ' STOKEN6 '
}


def tokenize_java(s, keep_comments=False):
    try:
        tokens = []
        assert isinstance(s, str)
        s = s.replace(r'\r', '')
        tokens_generator = javalang_tok.tokenize(
            s, keep_comments=keep_comments)
        for token in tokens_generator:
            if isinstance(token, javalang_tok.String):
                tokens.append(process_string(
                    token.value, JAVA_CHAR2TOKEN, JAVA_TOKEN2CHAR, False))
            elif isinstance(token, javalang_tok.Comment):
                com = process_string(
                    token.value, JAVA_CHAR2TOKEN, JAVA_TOKEN2CHAR, True)
                if len(com) > 0:
                    tokens.append(com)
            else:
                tokens.append(token.value)
        return tokens
    except:
        return []


def process_string(tok, char2tok, tok2char, is_comment):
    if is_comment:
        tok = re.sub(' +', ' ', tok)
        tok = re.sub(r"(.)\1\1\1\1+", r"\1\1\1\1\1", tok)
        if len(re.sub(r'\W', '', tok)) < 2:
            return ''
    tok = tok.replace(' ', ' ▁ ')
    for char, special_token in char2tok.items():
        tok = tok.replace(char, special_token)
    if tok.startswith(' STOKEN0'):
        if tok.endswith('\n'):
            tok = tok[:-1]
        tok += ' ENDCOM'
    tok = tok.replace('\n', ' STRNEWLINE ')
    tok = tok.replace('\t', ' TABSYMBOL ')
    tok = re.sub(' +', ' ', tok)
    # tok = tokenize_v14_international(tok)
    tok = TOKENIZERS['intl'](tok)
    tok = re.sub(' +', ' ', tok)
    for special_token, char in tok2char.items():
        tok = tok.replace(special_token, char)
    tok = tok.replace('\r', '')

    return tok


def tokenize_python(s, keep_comments=False):
    try:
        assert isinstance(s, str)
        s = s.replace(r'\r', '')
        tokens = []

        try:
            iterator = tokenize.tokenize(BytesIO(s.encode('utf-8')).readline)
        except SyntaxError as excep:
            raise SyntaxError(excep)

        removed_docstr = 0
        while True:
            try:
                toktype, tok, _, _, line = next(iterator)
            except (tokenize.TokenError, IndentationError, SyntaxError, UnicodeDecodeError):
                raise Exception(
                    f"Impossible to parse tokens because icorrect source code \"{s[0:30]}\" ...")
            except StopIteration:
                raise Exception(f"End of iterator before ENDMARKER token.")

            if toktype == tokenize.ENCODING or toktype == tokenize.NL:
                continue

            elif toktype == tokenize.NEWLINE:
                if removed_docstr == 1:
                    removed_docstr = 0
                    continue
                tokens.append('NEW_LINE')

            elif toktype == tokenize.COMMENT:
                if keep_comments:
                    com = process_string(
                        tok, PYTHON_CHAR2TOKEN, PYTHON_TOKEN2CHAR, True)
                    if len(com) > 0:
                        tokens.append(com)
                else:
                    continue

            elif toktype == tokenize.STRING:
                if tok == line.strip():  # docstring
                    if not keep_comments:
                        removed_docstr = 1
                        continue
                    else:
                        coms = process_string(
                            tok, PYTHON_CHAR2TOKEN, PYTHON_TOKEN2CHAR, True)
                        if len(coms) > 0:
                            tokens.append(coms)
                        else:
                            removed_docstr = 1
                else:
                    tokens.append(process_string(
                        tok, PYTHON_CHAR2TOKEN, PYTHON_TOKEN2CHAR, False))

            elif toktype == tokenize.INDENT:
                tokens.append('INDENT')

            elif toktype == tokenize.DEDENT:
                # empty block
                if tokens[-1] == 'INDENT':
                    tokens = tokens[:-1]
                else:
                    tokens.append('DEDENT')

            elif toktype == tokenize.ENDMARKER:
                tokens.append('ENDMARKER')
                break

            else:
                tokens.append(tok)

        assert (tokens[-1] == 'ENDMARKER'), "Error, no end marker"
        return tokens[:-1]
    except KeyboardInterrupt:
        raise
    except:
        return []