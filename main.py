from typing import List, Tuple, Optional, Dict, Union, Set, Deque

from collections import deque
import sqlparse
from sqlparse.sql import Token
from sqlparse.tokens import Whitespace, Newline, Punctuation, Comment, _TokenType, DML, Keyword, String, Name

# "Token-comparison" type: either a token-type or token-type/expected-body combo. ie: `Keyword` or (`Keyword`,'JOIN')
_TC = Union[_TokenType, Tuple[_TokenType, Optional[str]]]


def normalize_spec(spec: _TC) -> Tuple[_TokenType, Optional[str]]:
    """
    Normalizes the token-specification into a TYPE/BODY 2-tuple.

    :param spec: either a TYPE or a TYPE/BODY specification; the former gets a ``None`` body

    :return: TYPE/BODY specification
    """

    # If the spec is merely a TYPE (no BODY), result is TYPE/NONE
    if isinstance(spec, _TokenType):
        result = (spec, None)
    # Otherwise...
    else:
        # ... expect a TYPE/BODY specification; unpack those two objects
        try:
            specified_type, specified_body = spec
        except (TypeError, ValueError):
            # Couldn't unpack into two objects :(
            raise \
                ValueError(
                    f"Expecting 2-tuple for token-type specification, but got {type(spec).__name__} ({repr(spec)})"
                )

        # We have two objects, but are they a TYPE and a BODY?
        if isinstance(specified_type, _TokenType) and (specified_body is None or isinstance(specified_body, str)):
            result = (specified_type, specified_body)
        else:
            raise \
                ValueError(
                    f"Expecting <_TokenType,str> 2-tuple for specification, but got "
                    f"<{type(specified_type).__name__},{type(specified_body).__name__}>"
                )

    # Return the normalized type-specification
    return result


class Selection:
    """
    Data-structure containing data about a column being selected in a SQL query
    """
    def __init__(self, name: str, expression_tokens: List[Token]):
        self.name = name
        self.expression_tokens = expression_tokens

    @property
    def expression(self) -> str:
        """
        Gives the SQL formula defining the value for this selection.

        :return: SQL formula
        """
        substrings = []  # type: List[str]

        # Ensure no whitespace tokens participate in the front or back of the expression
        participating_tokens = trim(list(self.expression_tokens))

        i = 0
        while i < len(participating_tokens):
            # If the token at this index isn't whitespace, append its body to the substrings participating in the
            # expression
            if not check_token(participating_tokens, i, Whitespace, Newline):
                substrings.append(str(participating_tokens[i]))
                i += 1
            # Otherwise, the token at this index is whitespace
            else:
                # Append a single space to the substrings comprising the expression
                substrings.append(" ")

                # We don't want multiple spaces
                whitespace_end = i+1
                while whitespace_end < len(participating_tokens):
                    if not check_token(participating_tokens, whitespace_end, Whitespace, Newline):
                        break
                    whitespace_end += 1
                i = whitespace_end

        expression = ''.join(substrings)
        return expression

    def __str__(self):
        return f"{self.name}: {''.join(str(t) for t in self.expression_tokens)}"


class SQLTokenizer:
    """
    Utility class for converting SQL source-strings into a sequence of tokens/grammar-trees which describe the source.
    """

    @staticmethod
    def parse(source: str, flatten: bool = False) -> List[Token]:
        """
        Parses the SQL source into a sequence of tokens, where each token is a node in the SQL grammar tree.

        :param source: SQL-source to tokenize
        :param flatten: TRUE to return the sequence of leaf-node tokens; otherwise (when FALSE), returns tokens/
          grammar-nodes which may be composite (containing tokens within themselves)

        :return: sequence of grammar-tree nodes
        """

        # Before using the library's parser, replace some of the regexes in its SQL_REGEX list; that list stores
        # 2-tuples mapping a regex to the respective token-type which the regex describes
        from sqlparse.keywords import SQL_REGEX
        from sqlparse.tokens import Comment
        replacement_regex_by_token_type = \
            {
                # Introduce a `?` after the space following the pound-sign `#`: we want to indicate that the space is
                # optional... not *REQUIRED*
                Comment.Single: r'(--|# ?).*?(\r\n|\r|\n|$)',
                Comment.Single.Hint: r'(--|# ?)\+.*?(\r\n|\r|\n|$)',    # introduce a `?` follow
            }  # type: Dict[_TokenType, str]

        # Iterate over the 2-tuples in the list to determine whether the respective tuple requires replacement
        for i, regex_and_token in enumerate(SQL_REGEX):
            # Unpack the regex/token-type pair
            _, token_type = regex_and_token

            # If this token-type has a replacement, then replace this list element with one which we want
            replacement_regex = replacement_regex_by_token_type.get(token_type)
            if replacement_regex is not None:
                SQL_REGEX[i] = (replacement_regex, token_type)

        # Now that the parser has any revised regexes, run the parser to "tokenize" the SQL-source
        parsed_statements = sqlparse.parse(source)
        leading_statement = parsed_statements[0]
        statement_tokens = leading_statement.tokens

        # The library's "tokens" are actually nodes within a token-tree that it has built; if the user wants the
        # sequence of leaf-nodes, then "flatten" the tree (DFS exploration to return those nodes which are leaves of the
        # tree)
        if not flatten:
            result = statement_tokens  # type: List[Token]
        else:
            result = []
            for token in statement_tokens:
                result.extend(SQLTokenizer.expand(token))

        return result

    @staticmethod
    def expand(node: Token, _result: List[Token] = None) -> List[Token]:
        """
        Gives the sequence of atomic token (leaf nodes of the token tree) represented by the given token (arbitrary node
        within the token tree).

        :param node: node from which to search for tokens
        :param _result: result-sequence to which to append leaf nodes; a new list is created if this is not populated

        :return: token sequence
        """

        if _result is None:
            _result = []

        # BASE CASE: the node is a leaf node (no inner-tokens contained by this node) - append the token
        if not node.is_group:
            _result.append(node)

        # RECURSIVE CASE:
        else:
            # The token is not atomic: it is composed of child tokens... DFS recursion to reach the leaves
            for inner_token in node.flatten():
                SQLTokenizer.expand(inner_token, _result=_result)

        return _result

    @staticmethod
    def flatten(tokens: List[Token], start: int = 0, stop: int = None, filter_: Set[_TC] = None) -> List[Token]:
        """
        Returns a copy of the input-sequence having all composite tokens replaced by their atomic components.

        :param tokens: token-sequence to flatten (this sequence is unmodified)
        :param start: index at which to begin flattening
        :param stop: index (exclusive) at which to stop flattening
        :param filter_: set of token-specifications describing tokens to exclude from the flattened index

        :return: copy of the source range having all composite token exploded into their atomic components
        """

        # If no stopping point was defined, use the end of the list
        if stop is None:
            stop = len(tokens)

        # Ensure any filtering specifications are unpacked into a TYPE/BODY pair
        filter_specifications = set()  # type: Set[Tuple[_TokenType, Optional[str]]]
        if filter_ is not None:
            for filter_specification in filter_:
                filter_specifications.add(
                    normalize_spec(filter_specification)
                )

        # Begin building the result-sequence from the range specified from the source-sequence
        result = []  # type: List[Token]
        i = start
        while i < stop:
            # Resolve the next token to flatten from the source, then flatten it
            next_token = tokens[i]
            child_tokens = SQLTokenizer.expand(next_token)

            # Resolve those leaf-nodes which qualify for entry into the result (this would be all those which don't get
            # filtered out)
            if not filter_:
                qualifying_children = child_tokens  # type: List[Token]
            else:
                # Build the list of leaf-nodes which qualify (those which aren't getting filtered out)
                qualifying_children = []
                for child in child_tokens:
                    # Assume this leaf-node qualifies, but disqualify it if it matches one of the filter specs
                    qualifies = True
                    for illegal_type, specified_body in filter_specifications:
                        if match(child, illegal_type, body=specified_body):
                            qualifies = False
                            break

                    # If the node still qualifies, add it to the qualifying nodes
                    if qualifies:
                        qualifying_children.append(child)

            # Add the qualifying entrants to the list, then proceed to the next position from the source-sequence
            result.extend(qualifying_children)
            i += 1

        return result


def match(t: Token, control_type: _TokenType, body: str = None, cased: bool = False) -> bool:
    """
    Gives whether token matches the given type-criteria.

    :param t: token being scrutinized
    :param control_type: type which the token must match
    :param body: content to be matched by the token
    :param cased: (ignored when no ``body`` is specified) TRUE if the content is case-sensitive, false otherwise

    :return: TRUE if the token is described by the criteria, FALSE otherwise
    """
    if t.ttype != control_type:
        matches = False
    elif body is None:
        matches = True
    elif cased:
        matches = (str(t) == body)
    else:
        matches = (str(t).upper() == body.upper())
    return matches


def seek(
        tokens: List[Token], search_criteria: _TC, start: int = 0, stop: int = None, skip_for: _TC = None
) -> Optional[int]:
    """
    Gives the next index at which the specified token-type may be found.

    :param tokens: sequence of token through which to search
    :param search_criteria: type-specification describing the kind of token being sought
    :param start: index (inclusive) at which to begin searching for token
    :param stop: index at which to stop searching for token (the token at this index is not evaluated)
    :param skip_for: token-specification defining those tokens which, upon encounter, imply that what would otherwise be
      the result should be skipped over; this is useful in situations such as finding the close-parenthesis associated
      with a particular open-parenthesis: for each new ("inner") open-paren encountered during the search, it's
      appropriate to skip the next close-paren which would otherwise be the result

    :return: index at which to find the token (or ``None`` if no such token is available)
    """

    # Do some initialization
    result = None  # type: Optional[int]
    if stop is None:
        stop = len(tokens)

    # Resolve the type and body specifications defining what we're looking for
    subject_type, subject_body = normalize_spec(search_criteria)

    # Resolve the type and body specifications defining what prompts us to skip what would otherwise be a match
    if skip_for is None:
        skip_type = None    # type: Optional[_TokenType]
        skip_body = None    # type: Optional[str]
    else:
        skip_type, skip_body = normalize_spec(skip_for)

    # Begin iterating through the tokens, looking for that which we want
    i = start
    skip_counter = 0
    while i < stop:
        next_token = tokens[i]

        # If this token is the type we're looking for...
        if match(next_token, subject_type, body=subject_body):
            # ... we *probably* want to report this index; that is, unless we have to skip it :(
            if skip_counter > 0:
                skip_counter -= 1  # we're skipping this one... one less to skip at this point
            else:
                result = i
                break  # found what we're looking for!

        # Otherwise, this token is not the type we're looking for; if it's a type which signals that we should skip the
        # next instance of what we're looking for, increment the skip-counter
        elif skip_for is not None and match(next_token, skip_type, body=skip_body):
            skip_counter += 1

        i += 1

    return result


def check_token(tokens: List[Token], index: int, *types: _TC) -> bool:
    """
    Accesses the token at the given index to determine whether it is among the given type-specifications.

    **NOTE:** out-of-range index results in a ``False`` return rather than any error.

    :param tokens: sequence of tokens from which to sample
    :param index: position (0-based) of token to be sampled
    :param types: type-specifications which the sampled token may match

    :return: TRUE if the sampled token is found to match at least one of the type-specifications, FALSE otherwise
    """

    # Assume that no token at the given index will match any of the given specifications
    result = False

    # Try getting a token at the specified index
    try:
        token = tokens[index]
    except IndexError:
        pass  # Oh well: no token, no match
    else:
        # If there are any specifications to match, iterate through to find one which does
        if len(types) > 0:
            for type_specification in types:
                token_type, token_body = normalize_spec(type_specification)
                if match(token, token_type, body=token_body):
                    result = True
                    break

    return result


def check_non_whitespace(tokens: List[Token], index: int, *types: _TC) -> bool:
    """
    TODO EXPLAIN

    :param tokens:
    :param index:
    :param types:

    :return:
    """

    result = False

    tokens = [t for t in tokens if not (match(t, Whitespace) or match(t, Newline))]
    try:
        token = tokens[index]
    except IndexError:
        pass
    else:
        for type_spec in types:
            type_code, token_body = normalize_spec(type_spec)
            if match(token, type_code, body=token_body):
                result = True
                break

    return result


def trim(tokens: List[Token]) -> List[Token]:
    """
    Alters the input-list by removing any whitespace-tokens from its ends.

    :param tokens: sequence undergoing alteration

    :return: (for convenience) returns another reference to the altered input-list
    """

    while check_token(tokens, -1, Whitespace, Newline):
        tokens.pop()
    while check_token(tokens, 0, Whitespace, Newline):
        tokens.pop(0)
    return tokens


def parse_selection(tokens: List[Token]) -> Selection:
    """
    Reads a sequence of tokens representing a column selection to produce a ``Selection`` instance describing the
    selection.

    :param tokens: token-sequence representing a column selection

    :return: object describing the selection
    """

    # Remove any padding whitespace tokens
    trim(tokens)
    if len(tokens) == 0:
        raise ValueError("No substantive tokens")

    # Resolve the column-name from the final token: it really ought ot be either an identifier or a string-literal
    name_token = tokens[-1]
    if match(name_token, Name) or match(name_token, Name.Builtin) or match(name_token, Keyword):
        name = str(name_token)
    elif match(name_token, String) or match(name_token, String.Single) or match(name_token, String.Symbol):
        name = str(name_token)[1:-1]
    else:
        raise ValueError(f"Expecting a name to conclude the selection, but got token {repr(name_token)}")

    # If there's only one token, it's both the expression and the column-name
    if len(tokens) == 1:
        expression_tokens = tokens
    # Otherwise, there's multiple tokens participating in this selection
    else:
        # We have to determine whether these multiple tokens represent an expression with an implicit column name, or
        # instead whether the column-name is explicit: an alias applied following the expression; we know there's an
        # alias if the second-to-last token is the keyword "AS"
        if check_non_whitespace(tokens, -2, (Keyword, "AS")):
            expression_tokens = \
                trim(trim(tokens[:-1])[:-1])  # trim the last non-whitespace, and the last non-whitespace after that
        # Otherwise, there wasn't an "AS"; let's still treat it as an explicit alias if we find a string
        elif check_token(tokens, -1, String, String.Single, String.Symbol):
            expression_tokens = trim(tokens[:-1])

        # Otherwise, there's neither an "AS" nor even a string on the end; the only way this expression has an explicit
        # alias is if there's an identifier following the expression... we'll know that the identifier isn't part of the
        # expression if it doesn't have a period in front of it
        elif check_token(tokens, -1, Name, Name.Builtin, Keyword) and not check_non_whitespace(tokens, -2, (Punctuation, ".")):
            expression_tokens = trim(trim(tokens[:-1])[:-1])

        # Otherwise, there's no tokens to trim: the whole thing is the expression
        else:
            expression_tokens = tokens

    trim(expression_tokens)
    if check_token(expression_tokens, -1, Punctuation) and not check_token(expression_tokens, -1, (Punctuation, ")")):
        raise \
            ValueError(
                f"SQL expression not expected to conclude in punctuation, but final token of {repr(name)} field found "
                f"to be punctuation {repr(str(expression_tokens[-1]))}"
            )
    result = Selection(name, expression_tokens)
    return result


def parse_outputs(source: str) -> Dict[str, str]:
    """
    For the given SQL-grammar source code, gives a mapping of the names of columns to the expression formulating the
    value for the respective column.

    :param source: SQL-grammar source code to be parsed

    :return: column-names and the SQL-expression associated therewith
    """

    # Convert the SQL source into a sequence of tokens
    grammar_tree_nodes = SQLTokenizer.parse(source)

    # We expect to find a DML-type token expressing "SELECT", followed (though not *immediately*) by a Keyword-type
    # token expressing "FROM", between which ought to be the column-definitions we're seeking...
    # Find the `SELECT` and `FROM` tokens!
    select_index = seek(grammar_tree_nodes, (DML, "SELECT"))
    if select_index is None:
        raise ValueError("No SELECT token found in source")
    from_index = seek(grammar_tree_nodes, (Keyword, "FROM"), start=select_index+1)
    if from_index is None:
        raise ValueError("No FROM token found in source")

    # Resolve the tokens between SELECT and FROM
    column_content = \
        SQLTokenizer.flatten(
            grammar_tree_nodes, start=select_index+1, stop=from_index, filter_={Comment}
        )

    # Iterate over the tokens to find the indexes of those commas which separate one selection from another
    separating_commas = []  # type: List[int]
    opened_parenthesis = 0
    for i, token in enumerate(column_content):
        if opened_parenthesis == 0 and match(token, Punctuation, body=","):
            separating_commas.append(i)
        elif match(token, Punctuation, body="("):
            opened_parenthesis += 1
        elif match(token, Punctuation, body=")"):
            if opened_parenthesis == 0:
                raise ValueError("Encountered parenthesis-close, but no prior matching open-paren was parsed")
            opened_parenthesis -= 1
    if opened_parenthesis > 0:
        raise ValueError("Not all opened parenthesis were closed")

    # We know which among our sequence of tokens are commas separating the selections; parse the selections, consuming
    selections = deque()  # type: Deque[Selection]
    while separating_commas:
        next_separation_index = separating_commas.pop()
        next_sequence = column_content[next_separation_index+1:]
        selections.append(parse_selection(next_sequence))

        column_content = column_content[:next_separation_index]
    selections.append(parse_selection(column_content))

    # Iterate over the selections to build a COLUMN->FORMULA mapping for return
    expressions_by_output_name = {}  # type: Dict[str, str]
    for selection in selections:
        if selection.name in expressions_by_output_name:
            raise ValueError(f"Encountered duplicate outputs {repr(selection.name)}")
        expressions_by_output_name[selection.name] = selection.expression
    return expressions_by_output_name


def main():
    output_formulae_by_name_by_query = {}  # type: Dict[str, Dict[str, str]]
    import json
    with open("response_1715950277130.json") as infile:
        query_configs = json.load(infile)
    total = len(query_configs)
    successes = 0
    for query_config in query_configs:
        query_name = query_config["queryName"]      # type: str
        source = query_config["querySQLStream"]     # type: str
        try:
            # Parse the source to get our COLUMN->FORMULA pairs
            output_formulae_by_name = parse_outputs(source)
        except Exception as error:
            print(f"Error parsing query {repr(query_name)}: ({type(error).__name__}) {error}")
        else:
            output_formulae_by_name_by_query[query_name] = output_formulae_by_name
            for output_name, formula in output_formulae_by_name.items():
                expression = f"{output_name.rjust(30)} | {formula}"
                #outfile.write(expression + '\n')
                print(expression)

            successes += 1

    print(f"Successfully parsed {successes} of {total} queries")


if __name__ == "__main__":
    main()
    # TODO: overhaul token comparison
