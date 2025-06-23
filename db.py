from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost:5432/perceivenow_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text)
    answer = Column(Text)
    sources = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def log_query(query: str, answer: str, sources: list[str]):
    session = SessionLocal()
    log_entry = QueryLog(
        query=query,
        answer=answer,
        sources=", ".join(sources),
    )
    session.add(log_entry)
    session.commit()
    session.close()