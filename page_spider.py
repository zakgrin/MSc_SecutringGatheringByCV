import os
import argparse
from utilities import url_utilities, database_utilities


def main(database: str, url_list_file: str):
    big_word_list = []
    print("wer are going to work with " + database)
    print("wer are going to scan " + url_list_file)
    urls = url_utilities.load_urls_from_file(url_list_file)
    for url in urls:
        print("reading " + url)
        page_content = url_utilities.load_page(url=url)
        words = url_utilities.scrape_page(page_contents=page_content)
        big_word_list.extend(words)

    # database code
    # 1: manage the issue of path to be multi platform
    os.chdir(os.path.dirname(__file__))
    path = os.path.join(os.getcwd(), "words.db")
    # 2: create the database
    database_utilities.create_database(database_path=path)
    # 3: save list of words into the database
    database_utilities.save_words_to_data_base(database_path=path, words_list=big_word_list)
    # 4: report top 10 words by count
    top_list = database_utilities.print_database_table(database_path=path)
    print('Top 10 words:')
    for item in top_list:
        print(item)


if __name__ == "__main__":
    # Setup the parser with two arguments (from the command line)
    parser = argparse.ArgumentParser()
    parser.add_argument("-db", "--database", help="SQLite File Name")
    parser.add_argument("-i", "--input", help="File containing urls to read")
    # Does the parsing
    args = parser.parse_args()
    # Pull args out to standard variables (that can used in python)
    database_file = args.database
    input_file = args.input
    # Pass arguments to the main function
    main(database=database_file, url_list_file=input_file)
