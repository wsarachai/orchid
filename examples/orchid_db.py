import os
import io
import MySQLdb
import json
import base64
import glob
import StringIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import MySQLdb
import shutil

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

HOST = 'localhost'
USER = 'root'
PASS = '1234'

ROOT_DIR = '/Users/sarachaii/Desktop/orchid'

datas = []

def findOrchid(id):
    for o in datas:
        if o['orchid_id'] == id:
            return o


def copy_db_data():
    connection1=MySQLdb.connect(host=HOST,user=USER,passwd=PASS,db='orchid_db', charset='utf8')
    cursor=connection1.cursor()

    sql='SELECT DISTINCT' \
        ' o.orchid_id,' \
        ' o.genus,' \
        ' o.specie,' \
        ' o.thai_name,' \
        ' o.df_herb,' \
        ' o.df_smell,' \
        ' o.st_rare,' \
        ' o.st_endangered,' \
        ' o.st_native' \
        ' FROM orchid_tbl as o'

    cursor.execute(sql)
    data=cursor.fetchall()

    datas = []

    for r in data:
        data = {
            'orchid_id': r[0],
            'genus': r[1],
            'specie': r[2],
            'thai_name': r[3],
            'df_herb': r[4],
            'df_smell': r[5],
            'st_rare': r[6],
            'st_endangered': r[7],
            'st_native': r[8],
            'desc': ""
        }

        datas.append(data)

    connection1.close()

    connection2 = MySQLdb.connect(host=HOST, user=USER, passwd=PASS, db='rspg_orchid_db', charset='utf8')
    cursor = connection2.cursor()

    for d in datas:

        sql="UPDATE Specie as s SET s.endangered=0x0{0}, s.herb=0x0{1}, s.nativePlant=0x0{2}, s.rare=0x0{3}, s.smell=0x0{4} WHERE s.specieCode = '{5}'"
        sql = sql.format(d["st_endangered"], d["df_herb"], d["st_native"], d["st_rare"], d["df_smell"], d["orchid_id"])

        cursor.execute(sql)

    connection2.commit()
    connection2.close()

def write_data():
    connection=MySQLdb.connect(host=HOST,user=USER,passwd=PASS,db='orchid_db', charset='utf8')
    cursor=connection.cursor()

    sql='SELECT DISTINCT' \
        ' o.orchid_id,' \
        ' o.genus,' \
        ' o.specie,' \
        ' o.thai_name,' \
        ' o.df_herb,' \
        ' o.df_smell,' \
        ' o.st_rare,' \
        ' o.st_endangered,' \
        ' o.st_native' \
        ' FROM orchid_tbl as o'

    cursor.execute(sql)
    data=cursor.fetchall()

    for r in data:
        genus_dir = os.path.join(ROOT_DIR, r[1])
        if not os.path.isdir(genus_dir):
            os.mkdir(genus_dir)
        specie_dir = os.path.join(genus_dir, r[0] + '@' + r[2])

        if not os.path.isdir(specie_dir):
            os.mkdir(specie_dir)

        data = {
            'orchid_id': r[0],
            'genus': r[1],
            'specie': r[2],
            'thai_name': r[3],
            'df_herb': r[4],
            'df_smell': r[5],
            'st_rare': r[6],
            'st_endangered': r[7],
            'st_native': r[8],
            'desc': ""
        }

        desc_file = os.path.join(specie_dir, 'desc.json')
        thai_name = os.path.join(specie_dir, r[3])

        with io.open(thai_name, 'w', encoding='utf8') as outfile:
            outfile.write(to_unicode(r[3]))

        # Write JSON file
        with io.open(desc_file, 'w', encoding='utf8') as outfile:
            str_ = json.dumps(data,
                              indent=4, sort_keys=True,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))


        # Read JSON file
        with io.open(desc_file, encoding='utf8') as data_file:
            data_loaded = json.load(data_file)

        print(data == data_loaded)
        datas.append(data)


def getById(line):
    idx = line.index('RSPG-')
    return findOrchid(line[idx:].strip())


def read_raw_text():
    connection=MySQLdb.connect(host=HOST,user=USER,passwd=PASS,db='orchid_db', charset='utf8')
    cursor=connection.cursor()

    with open('/Users/sarachaii/Desktop/orchid_data.txt') as f:
        content = f.readlines()

        id = None
        for line in content:
            if 'RSPG-' in line:
                id = getById(line)

            if id != None:
                s = ''
                s = '\xe0\xb8\xa5\xe0\xb8\xb1\xe0\xb8\x81\xe0\xb8\xa9\xe0\xb8\x93\xe0\xb8\xb0\xe0\xb8\x97\xe0\xb8\xb2\xe0\xb8\x87\xe0\xb8\x9e\xe0\xb8\xa4\xe0\xb8\x81\xe0\xb8\xa9\xe0\xb8\xa8\xe0\xb8\xb2\xe0\xb8\xaa\xe0\xb8\x95\xe0\xb8\xa3\xe0\xb9\x8c'
                if s in line:
                    id['desc'] = line.decode('utf8')

                    genus_dir = os.path.join(ROOT_DIR, id['genus'])
                    specie_dir = os.path.join(genus_dir, )
                    specie_dir = os.path.join(genus_dir, id['orchid_id'] + '@' + id['specie'])
                    desc_file = os.path.join(specie_dir, 'desc.json')

                    sql = "UPDATE `orchid_tbl` SET `description` = '" + id['desc'] + "' WHERE `orchid_id` = '"+ id['orchid_id'] +"'"

                    cursor.execute(sql)

                    try:
                        # Write JSON file
                        with io.open(desc_file, 'w', encoding='utf8') as outfile:
                            str_ = json.dumps(id,
                                              indent=4, sort_keys=True,
                                              separators=(',', ': '), ensure_ascii=False)
                            outfile.write(to_unicode(str_))
                    except:
                        print (id['orchid_id'] + '-' + id['genus'] + '-' + id['specie'])
                        pass

                    id = None

        connection.commit()
        connection.close()



def imgToString(img_path):
    str = ""
    with open(img_path, "rb") as imageFile:
        str = base64.b64encode(imageFile.read())
    return str


def stringToImg(str):
    tempBuff = StringIO.StringIO()
    tempBuff.write(str.decode('base64'))
    tempBuff.seek(0)  # need to jump back to the beginning before handing it off to PIL
    return Image.open(tempBuff)


def getOrchidFromId(sid):
    orchids = []
    connection=MySQLdb.connect(host=HOST,user=USER,passwd=PASS,db='orchid_db', charset='utf8')
    cursor=connection.cursor()

    sql = "SELECT * FROM orchid_tbl as ord \
           WHERE ord.orchid_id = '%s'" % (sid)

    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            orchid = {
                'id': row[0],
                'orchid_id': row[1],
                'genus': row[2],
                'specie': row[3],
                'thai_name': row[4],
                'df_herb': row[5],
                'df_smell': row[6],
                'st_rare': row[7],
                'st_endangered': row[8],
                'st_native': row[9],
                'desc': row[10],
            }
            # Now print fetched result
            orchids.append(orchid)
    except:
        print "Error: unable to fecth data"

    connection.close()
    return orchids


def getImageFromFilename(filename):
    images = []
    connection=MySQLdb.connect(host=HOST,user=USER,passwd=PASS,db='orchid_db', charset='utf8')
    cursor=connection.cursor()

    sql = "SELECT * FROM image_tbl as img \
           WHERE img.name = '%s'" % (filename)

    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            image = {
                'id': row[0],
                'orchid_id': row[1],
                'desc': row[2],
                'img': row[3],
                'name': row[4]
            }
            # Now print fetched result
            images.append(image)
    except:
        print "Error: unable to fecth data"

    connection.close()
    return images


def isImageExist(filename):
    return len(getImageFromFilename(filename)) > 0


def insertImage(img):
    connection=MySQLdb.connect(host=HOST,user=USER,passwd=PASS,db='orchid_db', charset='utf8')
    cursor=connection.cursor()

    try:
        query = "INSERT INTO image_tbl (`orchid_id`, `name`, `description`, `img`) VALUES ({0},'{1}','{2}','{3}')"
        query = query.format(img['orchid_id'], img['name'], img['description'], img['img'])

        cursor.execute(query)
        connection.commit()
    except:
        connection.rollback()
        pass

    connection.close()


def updateImage(img):
    connection=MySQLdb.connect(host=HOST,user=USER,passwd=PASS,db='orchid_db', charset='utf8')
    cursor=connection.cursor()

    try:
        query = "UPDATE image_tbl as img SET img.img='{}' WHERE img.id=" + str(img['id'])
        query = query.format(MySQLdb.escape_string(img['img']))

        #cursor.execute(query, (MySQLdb.escape_string(img['img'])))
        cursor.execute(query)
        connection.commit()
    except:
        connection.rollback()
        pass

    connection.close()


def modifyDB():

    dirs = glob.glob("/Users/sarachaii/Desktop/orchid/*/")
    for d in dirs:
        sdirs = glob.glob(d + "/*/")
        for sd in sdirs:
            ssdirs = glob.glob(sd + "/*/")
            for ssd in ssdirs:
                files = glob.glob(os.path.join(ssd, "*.jpg"))
                for f in files:
                    sid = f[f.index('RSPG-'):]
                    oid = sid.index('@')
                    sid = sid[:oid]

                    orchid = getOrchidFromId(sid)[0]
                    filename = os.path.basename(f)

                    if not isImageExist(filename):
                        str = imgToString(f)

                        img = {
                            'orchid_id': orchid['id'],
                            'name': filename,
                            'description': '',
                            'img': str
                        }

                        insertImage(img)
                    else:
                        img = getImageFromFilename(filename)[0]
                        if not img['img']:
                            img['img'] = imgToString(f)
                            updateImage(img)


def modifyName():
    dirs = glob.glob("/Users/sarachaii/Desktop/orchid/*/")
    for d in dirs:
        sdirs = glob.glob(d + "/*/")
        for sd in sdirs:
            ssdirs = glob.glob(sd + "/*/")
            for ssd in ssdirs:
                files = glob.glob(os.path.join(ssd, "*.jpg"))
                fid = 1
                for f in files:
                    print f
                    stype = f[:f.rfind('/')]
                    stype = stype[stype.rfind('/')+1:].title()
                    sid = f[f.index('RSPG-'):]
                    oid = sid.index('@')
                    sid = sid[:oid]
                    name = '{}_{}_{:03d}{}'.format(sid,stype,fid,'.jpg')
                    fid += 1
                    di = f.rfind('/')
                    newd = f[:di]
                    newname = os.path.join(newd, name)
                    os.rename(f, newname)


def formatName(f, fid):
    sid = f[f.index('RSPG-'):]
    oid = sid.index('@')
    sid = sid[:oid]
    return '{}_{:03d}{}'.format(sid, fid, '.jpg')


def modifyName1():
    dirs = glob.glob("/Users/sarachaii/Desktop/orchid/*/")
    for d in dirs:
        sdirs = glob.glob(d + "/*/")
        for sd in sdirs:
            files = glob.glob(os.path.join(sd, "*.jpg"))
            imgDir = os.path.join(sd, "images")
            if not os.path.exists(imgDir):
                os.makedirs(imgDir)
            fid = 1
            maxfile = os.path.join(sd, ".maxfiles")
            if os.path.isfile(maxfile):
                with io.open(maxfile, 'r') as infile:
                    try:
                        fmax = infile.read();
                        fid = int(fmax)
                    except:
                        pass
            for f in files:
                print f
                newname = formatName(f, fid)
                dstpath = os.path.join(imgDir, newname)
                moveFile(f, dstpath)

                fid += 1
            with io.open(maxfile, 'w', encoding='utf8') as outfile:
                outfile.write(to_unicode(str(fid)))


def deleteDir(dir):
    try:
        shutil.rmtree(dir)
    except shutil.Error as e:
        print('Error: %s' % e)
    except IOError as e:
        print('Error: %s' % e.strerror)


def moveFile(src, dest):
    try:
        shutil.move(src, dest)
    # eg. src and dest are the same file
    except shutil.Error as e:
        print('Error: %s' % e)
    # eg. source or destination doesn't exist
    except IOError as e:
        print('Error: %s' % e.strerror)


def copyFile(src, dest):
    try:
        shutil.copy(src, dest)
    # eg. src and dest are the same file
    except shutil.Error as e:
        print('Error: %s' % e)
    # eg. source or destination doesn't exist
    except IOError as e:
        print('Error: %s' % e.strerror)


def showImg(img):
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    #img = mpimg.imread('/Users/sarachaii/Desktop/orchid/Dendrobium/RSPG-8001-34-100-001@secundum  (Bl.) Lindl./flowers/img-004.jpg')
    #showImg(img)
    #write_data()
    #read_raw_text()
    modifyName1()
    #modifyDB()
    #showImg()
    #str = imgToString('/Users/sarachaii/Desktop/orchid/Dendrobium/RSPG-8001-34-100-001@secundum  (Bl.) Lindl./flowers/img-001.jpg')
    #showImg(stringToImg(str))
    #copy_db_data()

