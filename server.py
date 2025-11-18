from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from dateutil import tz
from sqlalchemy import create_engine, Column, Integer, String, DateTime, select, func, and_
from sqlalchemy.orm import declarative_base, sessionmaker
from fastapi.middleware.cors import CORSMiddleware

# ===== DB setup =====
engine = create_engine("sqlite:///parking.db", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    plate = Column(String, index=True)
    ts = Column(DateTime, index=True)
    status = Column(String, index=True)  # "IN"/"OUT"

Base.metadata.create_all(engine)

# ===== FastAPI =====
app = FastAPI(title="Smart Parking API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ===== Models =====
class EventIn(BaseModel):
    plate: str
    ts: Optional[datetime] = None

class EventOut(BaseModel):
    id: int
    plate: str
    ts: datetime
    status: str

# ===== Helpers =====
TZ_BKK = tz.gettz("Asia/Bangkok")
TZ_UTC = tz.gettz("UTC")

def now_utc():
    return datetime.now(tz=TZ_BKK).astimezone(TZ_UTC)

def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ_BKK)
    return dt.astimezone(TZ_UTC)

def to_bkk(dt: datetime) -> datetime:
    return dt.astimezone(TZ_BKK)

# ===== API =====
@app.post("/events", response_model=EventOut)
def create_event(payload: EventIn):
    plate = payload.plate.strip().upper()
    if not plate:
        raise ValueError("plate rỗng")

    with SessionLocal() as db:
        cnt = db.execute(select(func.count()).select_from(Event).where(Event.plate == plate)).scalar_one()
        status = "IN" if cnt % 2 == 0 else "OUT"
        ts_utc = to_utc(payload.ts) if payload.ts else now_utc()

        ev = Event(plate=plate, ts=ts_utc, status=status)
        db.add(ev)
        db.commit()
        db.refresh(ev)
        return EventOut(id=ev.id, plate=ev.plate, ts=to_bkk(ev.ts), status=ev.status)

@app.get("/events", response_model=List[EventOut])
def list_events(
    plate: Optional[str] = Query(None),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=2000)
):
    with SessionLocal() as db:
        conds = []
        if plate:
            conds.append(Event.plate == plate.strip().upper())
        if date_from:
            conds.append(Event.ts >= to_utc(date_from))
        if date_to:
            conds.append(Event.ts <= to_utc(date_to))
        if status:
            conds.append(Event.status == status.strip().upper())

        stmt = select(Event).where(and_(*conds)) if conds else select(Event)
        stmt = stmt.order_by(Event.ts.desc()).limit(limit)
        rows = db.execute(stmt).scalars().all()

        return [EventOut(id=r.id, plate=r.plate, ts=to_bkk(r.ts), status=r.status) for r in rows]

@app.get("/plates/{plate}/last", response_model=Optional[EventOut])
def last_event(plate: str):
    with SessionLocal() as db:
        stmt = select(Event).where(Event.plate == plate.strip().upper()).order_by(Event.ts.desc()).limit(1)
        r = db.execute(stmt).scalar_one_or_none()
        if not r:
            return None
        return EventOut(id=r.id, plate=r.plate, ts=to_bkk(r.ts), status=r.status)
