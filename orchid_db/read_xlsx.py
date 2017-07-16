import openpyxl
import MySQLdb


class Orchid:
    count = 0

    def __init__(self, orchid_id, genus, specie, thai_name, df_herb=0, df_smell=0, st_rare=0, st_endangered=0, st_native=0):
        self.uid = 0
        self.orchid_id = orchid_id
        self.genus = genus
        self.specie = specie
        self.thai_name = thai_name
        self.df_herb = df_herb
        self.df_smell = df_smell
        self.st_rare = st_rare
        self.st_endangered = st_endangered
        self.st_native = st_native


def main():
    book = openpyxl.load_workbook( "/Users/sarachaii/Desktop/orchid_db.xlsx")

    sheet = book.active

    cells = sheet['A4': 'I947']
    orchids = []

    c = 0
    for c1, c2, c3, c4, c5, c6, c7, c8, c9 in cells:
        if (c1.value):
            obj = Orchid(c1.value, c2.value, c3.value, unicode(c4.value).encode('utf8'))

            if c5.value:
                obj.df_herb = 1
            if c6.value:
                obj.df_smell = 1
            if c7.value:
                obj.st_rare = 1
            if c8.value:
                obj.st_endangered = 1
            if c9.value:
                obj.st_native = 1

            orchids.append(obj)
            c += 1

    print ("Total {0}".format(c))

    db = MySQLdb.connect("localhost", "root", "1234", "orchid_db", charset='utf8')
    cursor = db.cursor()

    for obj in orchids:
        print("{0:s} {1:s} {2:s} {3:s}".format(obj.orchid_id, obj.genus, obj.specie, obj.thai_name))

        sql = "INSERT INTO `orchid_tbl` (`id`, `orchid_id`, `genus`, `specie`, `thai_name`, `df_herb`, `df_smell`, `st_rare`, `st_endangered`, `st_native`) "
        values = "VALUES(NULL, '{0:s}', '{1:s}', '{2:s}', '{3:s}', {4:d}, {5:d}, {6:d}, {7:d}, {8:d})"\
            .format(obj.orchid_id, obj.genus, obj.specie, obj.thai_name, obj.df_herb, obj.df_smell, obj.st_rare, obj.st_endangered, obj.st_native)

        sql += values

        print (sql)

        try:
            # Execute the SQL command
            cursor.execute(sql)
            # Commit your changes in the database
            db.commit()
        except:
            # Rollback in case there is any error
            db.rollback()

    # disconnect from server
    db.close()

if __name__ == '__main__':
    main()


