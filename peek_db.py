import sqlite3, datetime
from dateutil import tz

conn = sqlite3.connect("parking.db")
cur = conn.cursor()

print("Tables:")
for (name,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    print("-", name)

print("\nLatest 10 events:")
for row in cur.execute("SELECT id, plate, status, ts FROM events ORDER BY ts DESC LIMIT 10"):
    _id, plate, status, ts_utc = row
    # ts trong DB là UTC ISO/string; in thêm giờ Bangkok cho dễ nhìn
    from dateutil import parser
    bkk = parser.parse(ts_utc).astimezone(tz.gettz("Asia/Bangkok"))
    print(_id, plate, status, bkk.isoformat())

conn.close()
