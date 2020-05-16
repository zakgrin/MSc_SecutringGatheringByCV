import sqlite3 as lite


def create_database(database_path: str):
    conn = lite.connect(database_path)
    with conn:
        cur = conn.cursor()
        cur.execute("drop table if exists words")
        ddl = "CREATE TABLE words (word TEXT PRIMARY KEY NOT NULL, usage_count INT DEFAULT 1 NOT NULL);"
        cur.execute(ddl)
        ddl = "CREATE UNIQUE INDEX  words_word_uindex ON words (word)"
        cur.execute(ddl)
    conn.close()


def save_words_to_data_base(database_path: str, words_list: list):
    conn = lite.connect(database_path)
    with conn:
        cur = conn.cursor()
        for word in words_list:
            # check to see if the word is in there
            sql = "SELECT count(word) FROM words WHERE word='" + word + "' "
            cur.execute(sql)
            count = cur.fetchone()[0]
            if count > 0:
                sql = "UPDATE words SET usage_count = usage_count + 1 WHERE word = '" + word + "'"
            else:
                sql = "INSERT INTO words(word) values ('" + word + "')"
            cur.execute(sql)
    conn.close()
    print("Database operation is complete!")


def print_database_table(database_path: str):
    top_10_list = list()
    conn = lite.connect(database_path)
    with conn:
        cur = conn.cursor()
        sql = "SELECT * FROM words ORDER BY usage_count DESC"
        cur.execute(sql)
        for row in cur.fetchall()[:10]:
            # print(row)
            top_10_list.append(row)
    conn.close()
    # print('Top 10 rows of the database are printed!')
    return top_10_list