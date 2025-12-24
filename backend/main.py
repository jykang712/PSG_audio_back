# backend/main.py

from fastapi import FastAPI, UploadFile, File, status, HTTPException, Query
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import logging
import os
import shutil
from datetime import datetime, timedelta, timezone, time  
from typing import Optional

# import librosa
import pymysql

# ============================
# TF-Hub 캐시 디렉터리 설정
# ============================
TFHUB_CACHE_DIR = r"C:\tfhub_cache_psg"

if os.path.exists(TFHUB_CACHE_DIR) and not os.path.isdir(TFHUB_CACHE_DIR):
    os.remove(TFHUB_CACHE_DIR)

os.makedirs(TFHUB_CACHE_DIR, exist_ok=True)
os.environ["TFHUB_CACHE_DIR"] = TFHUB_CACHE_DIR

# ============================
# 기본 설정
# ============================

KST = timezone(timedelta(hours=9))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI 앱
app = FastAPI()

# CORS 설정 (프론트엔드)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# 1. 모델 로드 (Keras + YAMNet)
# ============================

# YAMNet TF-Hub 핸들
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
yamnet_model = None  # lazy loading

def get_yamnet_model():
    """
    TF-Hub에서 YAMNet 모델을 lazy-loading으로 불러오는 함수
    """
    global yamnet_model
    if yamnet_model is None:
        logger.info("Loading YAMNet model from TF-Hub...")
        yamnet_model = hub.load(YAMNET_HANDLE)
        logger.info("YAMNet loaded.")
    return yamnet_model

# YAMNet spectrogram을 입력으로 받는 Keras 모델 (best_model.keras)
MODEL_PATH = "../backend/models/best_model.keras"
logger.info(f"Loading Keras model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
logger.info("Keras model loaded.")

# 클래스 매핑 (0~4)
LABEL_MAP = {
    0: "Normal",             # 정상 호흡
    1: "Hypopnea",           # 저호흡
    2: "Mixed Apnea",        # 혼합성 무호흡
    3: "Obstructive Apnea",  # 폐쇄성 무호흡
    # 4: "Central Apnea",      # 중추성 무호흡
}

# ============================
# 비밀번호 해시 유틸 (bcrypt)
# ============================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# ============================
# Pydantic 스키마
# ============================

class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    birth_date: str   # "YYYY-MM-DD"
    gender: str       # 'Male' | 'Female' | 'Other'

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserSettingRequest(BaseModel):
    """
    온보딩 3화면(수면시간, 각성 체크 여부, 알람 여부)을
    한 번에 서버로 보내는 요청 바디
    """
    user_id: int
    sleep_time: time               # 예: "23:00" -> time(23, 0)
    wake_up_time: time             # 예: "07:00" -> time(7, 0)
    is_awake_check_enabled: bool
    is_alarm_enabled: bool

# [추가] 프로필 수정용 스키마
class UserProfileUpdate(BaseModel):
    """
    Settings 화면에서 프로필 수정 시 사용하는 요청 바디
    (프론트는 camelCase: birthDate)
    """
    name: str
    birthDate: Optional[str] = None  # "YYYY-MM-DD"
    gender: Optional[str] = None  # 'Male' | 'Female' | 'Other'


# [추가] 토글만 수정할 때 사용하는 스키마
class UserSettingToggleRequest(BaseModel):
    """
    Settings 화면에서 각성 알림 / 알람 토글만 변경할 때 사용하는 요청 바디
    (sleep_time, wake_up_time은 건드리지 않음)
    """
    is_awake_check_enabled: bool
    is_alarm_enabled: bool

# [추가] SleepSettings에서 취침/기상 시간만 수정할 때 사용하는 스키마
class SleepScheduleUpdate(BaseModel):
    sleep_time: time  # "23:00"
    wake_up_time: time  # "07:00"



# ============================
# 2. DB 연결 헬퍼
# ============================

def get_db_connection():
    """
    MySQL 연결 함수.
    MySQL 서버의 time_zone은 '+09:00' (KST)로 설정되어 있다고 가정.
    """
    return pymysql.connect(
        host="127.0.0.1",
        user="sleepapp",       # 실제 계정으로 변경 가능
        password="sleep1234",
        db="deepsleepDB",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def update_sleep_report(user_id: int, label: int):
    """
    Sleep_Report 테이블에 오늘자 리포트 row를 생성/업데이트한다.

    - report_date: CURDATE() (MySQL, KST 기준)
    - 첫 insert 시:
        start_analysis_time = NOW()
        end_analysis_time   = NOW()
        apnea_level_X       = 1 (해당 라벨만 1, 나머지 0)
    - 이미 존재한다면:
        end_analysis_time     = NOW()
        apnea_level_X         = apnea_level_X + 1
        sleep_analysis_time   = TIMEDIFF(NOW(), start_analysis_time)
    """
    if label not in [0, 1, 2, 3]:
        label = 0

    inc = [0, 0, 0, 0]
    inc[label] = 1
    inc0, inc1, inc2, inc3 = inc

    sql = """
    INSERT INTO sleep_report (
        user_id,
        report_date,
        start_analysis_time,
        end_analysis_time,
        sleep_analysis_time,
        apnea_level_0,
        apnea_level_1,
        apnea_level_2,
        apnea_level_3
    ) VALUES (
        %s,
        CURDATE(),
        NOW(),
        NOW(),
        NULL,
        %s, %s, %s, %s
    )
    ON DUPLICATE KEY UPDATE
        end_analysis_time = NOW(),
        apnea_level_0 = apnea_level_0 + VALUES(apnea_level_0),
        apnea_level_1 = apnea_level_1 + VALUES(apnea_level_1),
        apnea_level_2 = apnea_level_2 + VALUES(apnea_level_2),
        apnea_level_3 = apnea_level_3 + VALUES(apnea_level_3),
        sleep_analysis_time = TIMEDIFF(NOW(), start_analysis_time);
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, inc0, inc1, inc2, inc3))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_sleep_report(user_id: int, report_date: Optional[str] = None):
    """
    Sleep_Report에서 특정 user + date 리포트 한 건을 가져온다.
    report_date가 None이면 MySQL CURDATE() 기준 오늘 날짜 리포트.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if report_date is None:
                sql = """
                    SELECT *
                    FROM sleep_report
                    WHERE user_id = %s
                      AND report_date = CURDATE()
                    LIMIT 1;
                """
                cur.execute(sql, (user_id,))
            else:
                sql = """
                    SELECT *
                    FROM sleep_report
                    WHERE user_id = %s
                      AND report_date = %s
                    LIMIT 1;
                """
                cur.execute(sql, (user_id, report_date))
            row = cur.fetchone()
        return row
    finally:
        conn.close()

def upsert_user_setting(data: UserSettingRequest):
    """
    User_Setting 테이블에 유저 설정을 저장/수정하고,
    User.has_completed_onboarding 을 1로 업데이트한다.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 1) 사용자 존재 여부 확인
            cur.execute("SELECT 1 FROM user WHERE user_id = %s LIMIT 1", (data.user_id,))
            exists_user = cur.fetchone()
            if not exists_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"존재하지 않는 사용자 ID입니다: {data.user_id}",
                )

            # 2) User_Setting 존재 여부 확인
            cur.execute(
                "SELECT setting_id FROM user_setting WHERE user_id = %s LIMIT 1",
                (data.user_id,),
            )
            row = cur.fetchone()

            # datetime.time -> "HH:MM:SS" 문자열로 변환
            sleep_time_str = data.sleep_time.strftime("%H:%M:%S")
            wake_up_time_str = data.wake_up_time.strftime("%H:%M:%S")

            if not row:
                # INSERT
                insert_sql = """
                    INSERT INTO user_setting (
                        user_id,
                        sleep_time,
                        wake_up_time,
                        is_awake_check_enabled,
                        is_alarm_enabled,
                        created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, NOW()
                    )
                """
                cur.execute(
                    insert_sql,
                    (
                        data.user_id,
                        sleep_time_str,
                        wake_up_time_str,
                        int(data.is_awake_check_enabled),
                        int(data.is_alarm_enabled),
                    ),
                )
                logger.info(
                    "[upsert_user_setting] inserted user_setting for user_id=%s",
                    data.user_id,
                )
            else:
                # UPDATE
                update_sql = """
                    UPDATE user_setting
                    SET
                        sleep_time = %s,
                        wake_up_time = %s,
                        is_awake_check_enabled = %s,
                        is_alarm_enabled = %s,
                        created_at = NOW()  -- '설정 생성/변경 시각' 용도
                    WHERE user_id = %s
                """
                cur.execute(
                    update_sql,
                    (
                        sleep_time_str,
                        wake_up_time_str,
                        int(data.is_awake_check_enabled),
                        int(data.is_alarm_enabled),
                        data.user_id,
                    ),
                )
                logger.info(
                    "[upsert_user_setting] updated user_setting for user_id=%s",
                    data.user_id,
                )

            # 3) 온보딩 완료 플래그 업데이트
            cur.execute(
                """
                UPDATE user
                SET has_completed_onboarding = 1
                WHERE user_id = %s
                """,
                (data.user_id,),
            )
            logger.info(
                "[upsert_user_setting] set has_completed_onboarding=1 for user_id=%s",
                data.user_id,
            )

        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.error("[upsert_user_setting] error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"사용자 설정 저장 중 오류가 발생했습니다: {e}",
        )
    finally:
        conn.close()



# ============================
# 4. 회원가입 / 로그인 API
# ============================

@app.post("/api/signup")
async def api_signup(payload: SignupRequest):
    """
    회원가입:
    - 이메일 중복 체크
    - 비밀번호 bcrypt 해시 저장
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 이메일 중복 확인
            cur.execute("SELECT user_id FROM user WHERE email = %s", (payload.email,))
            exists = cur.fetchone()
            if exists:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="이미 가입된 이메일입니다."
                )

            hashed_pw = hash_password(payload.password)

            sql = """
                INSERT INTO user (name, birth_date, gender, email, password_hash)
                VALUES (%s, %s, %s, %s, %s)
            """
            cur.execute(
                sql,
                (payload.name, payload.birth_date, payload.gender, payload.email, hashed_pw)
            )
            user_id = cur.lastrowid
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"회원가입 처리 중 오류가 발생했습니다: {e}"
        )
    finally:
        conn.close()

    return {
        "message": "회원가입 성공",
        "user": {
            "user_id": user_id,
            "name": payload.name,
            "email": payload.email,
            "birth_date": payload.birth_date,
            "gender": payload.gender,
            "hasCompletedOnboarding": False,
        }
    }


@app.post("/api/login")
async def api_login(payload: LoginRequest):
    """
    로그인:
    - 이메일로 사용자 조회
    - bcrypt 비밀번호 검증
    - User.has_completed_onboarding 그대로 사용
    """
    logger.info("[api_login] called: email=%s", payload.email)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT
                    user_id,
                    name,
                    birth_date,
                    gender,
                    email,
                    password_hash,
                    has_completed_onboarding
                FROM user
                WHERE email = %s
                LIMIT 1
            """
            cur.execute(sql, (payload.email,))
            row = cur.fetchone()

            if not row:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="존재하지 않는 이메일입니다."
                )

            if not verify_password(payload.password, row["password_hash"]):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="비밀번호가 올바르지 않습니다."
                )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"로그인 처리 중 오류가 발생했습니다: {e}"
        )
    finally:
        conn.close()

    return {
        "message": "로그인 성공",
        "user": {
            "user_id": row["user_id"],
            "name": row["name"],
            "email": row["email"],
            "birth_date": row["birth_date"].isoformat() if row["birth_date"] else None,
            "gender": row["gender"],
            "hasCompletedOnboarding": bool(row["has_completed_onboarding"]),
        }
    }



# ============================
# 5. 프로필 / 설정 조회·수정 API
# ============================

# [추가] 사용자 프로필 조회
@app.get("/api/users/{user_id}/profile")
async def api_get_user_profile(user_id: int):
    """
    사용자 프로필 조회:
    - User 테이블에서 name, birth_date, gender, email 조회
    - 나이는 서버에서 계산해서 내려줌
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, name, birth_date, gender, email
                FROM user
                WHERE user_id = %s
                LIMIT 1
                """,
                (user_id,),
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"존재하지 않는 사용자 ID입니다: {user_id}",
            )

        birth = row["birth_date"]
        age = None
        if birth:
            today = datetime.now(KST).date()
            age = today.year - birth.year - (
                (today.month, today.day) < (birth.month, birth.day)
            )

        return {
            "user_id": row["user_id"],
            "name": row["name"],
            "email": row["email"],
            "birthDate": birth.strftime("%Y-%m-%d") if birth else None,
            "gender": row["gender"],
            "age": age,
        }

    finally:
        conn.close()


# [추가] 사용자 프로필 수정
@app.put("/api/users/{user_id}/profile")
async def api_update_user_profile(user_id: int, payload: UserProfileUpdate):
    """
    사용자 프로필 수정:
    - name, birth_date, gender 업데이트
    - 업데이트 후 최신 프로필을 다시 반환
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 사용자 존재 여부 확인
            cur.execute("SELECT 1 FROM user WHERE user_id = %s LIMIT 1", (user_id,))
            exists = cur.fetchone()
            if not exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"존재하지 않는 사용자 ID입니다: {user_id}",
                )

            update_sql = """
                UPDATE user
                SET
                    name = %s,
                    birth_date = %s,
                    gender = %s
                WHERE user_id = %s
            """
            cur.execute(
                update_sql,
                (
                    payload.name,
                    payload.birthDate,  # "YYYY-MM-DD" 또는 None
                    payload.gender,
                    user_id,
                ),
            )

        conn.commit()

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.error("[api_update_user_profile] error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"프로필 업데이트 중 오류가 발생했습니다: {e}",
        )
    finally:
        conn.close()

    # 업데이트 후 최신 값 다시 조회해서 반환
    return await api_get_user_profile(user_id)


# [추가] 설정 조회 (수면 시간 + 토글 상태)
@app.get("/api/users/{user_id}/settings")
async def api_get_user_settings(user_id: int):
    """
    Settings 화면에서 사용할 사용자 설정 조회:
    - User_Setting에서 sleep_time, wake_up_time, is_awake_check_enabled, is_alarm_enabled 조회
    - 없으면 기본값으로 응답
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sleep_time, wake_up_time, is_awake_check_enabled, is_alarm_enabled
                FROM user_setting
                WHERE user_id = %s
                LIMIT 1
                """,
                (user_id,),
            )
            row = cur.fetchone()

        if not row:
            # 아직 온보딩을 안 했거나 설정이 없을 때 기본값
            return {
                "user_id": user_id,
                "sleep_time": None,
                "wake_up_time": None,
                "is_awake_check_enabled": False,
                "is_alarm_enabled": False,
            }


        def fmt_time(t):
            if t is None:
                return None

            # pymysql이 TIME을 timedelta로 반환하는 경우 처리
            if isinstance(t, timedelta):
                total_seconds = int(t.total_seconds())
                total_seconds = total_seconds % (24 * 3600)   # 24시간 단위로 보정
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                return f"{hours:02d}:{minutes:02d}"

            # datetime.time 타입인 경우
            return t.strftime("%H:%M")

        return {
            "user_id": user_id,
            "sleep_time": fmt_time(row["sleep_time"]),
            "wake_up_time": fmt_time(row["wake_up_time"]),
            "is_awake_check_enabled": bool(row["is_awake_check_enabled"]),
            "is_alarm_enabled": bool(row["is_alarm_enabled"]),
        }

    finally:
        conn.close()



# [추가] 토글만 수정 (각성 알림 / 알람)
@app.put("/api/users/{user_id}/settings")
async def api_update_user_settings(user_id: int, payload: UserSettingToggleRequest):
    """
    Settings 화면에서 각성 알림 / 알람 on/off만 수정할 때 사용하는 API.
    - sleep_time, wake_up_time은 건드리지 않음 (온보딩 /api/user-setting 또는 sleep-schedule API에서 관리)
    - User_Setting row가 없으면 기본 시간(23:00/07:00) + 토글 값으로 생성
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # User 존재 여부 확인
            cur.execute("SELECT 1 FROM user WHERE user_id = %s LIMIT 1", (user_id,))
            exists_user = cur.fetchone()
            if not exists_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"존재하지 않는 사용자 ID입니다: {user_id}",
                )

            # User_Setting 존재 여부 확인
            cur.execute(
                "SELECT sleep_time, wake_up_time FROM user_setting WHERE user_id = %s LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()

            if not row:
                # 아직 설정이 없다면 기본 시간으로 INSERT
                insert_sql = """
                    INSERT INTO user_setting (
                        user_id,
                        sleep_time,
                        wake_up_time,
                        is_awake_check_enabled,
                        is_alarm_enabled,
                        created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, NOW()
                    )
                """
                cur.execute(
                    insert_sql,
                    (
                        user_id,
                        "23:00:00",
                        "07:00:00",
                        int(payload.is_awake_check_enabled),
                        int(payload.is_alarm_enabled),
                    ),
                )
            else:
                # 기존 row가 있으면 토글만 업데이트
                update_sql = """
                    UPDATE user_setting
                    SET
                        is_awake_check_enabled = %s,
                        is_alarm_enabled = %s
                    WHERE user_id = %s
                """
                cur.execute(
                    update_sql,
                    (
                        int(payload.is_awake_check_enabled),
                        int(payload.is_alarm_enabled),
                        user_id,
                    ),
                )

        conn.commit()

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.error("[api_update_user_settings] error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"사용자 설정 업데이트 중 오류가 발생했습니다: {e}",
        )
    finally:
        conn.close()

    # 업데이트 후 최신 설정 반환
    return await api_get_user_settings(user_id)


# [추가] 취침/기상 시간만 수정 (SleepSettings에서 사용)
@app.put("/api/users/{user_id}/sleep-schedule")
async def api_update_sleep_schedule(user_id: int, payload: SleepScheduleUpdate):
    """
    SleepSettings 화면에서 '취침 시간 / 기상 시간'을 수정할 때 사용하는 API.

    - is_awake_check_enabled, is_alarm_enabled 값은 기존 User_Setting 값 유지
      (없으면 기본값으로 INSERT)
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 1) User 존재 여부 확인
            cur.execute("SELECT 1 FROM user WHERE user_id = %s LIMIT 1", (user_id,))
            exists_user = cur.fetchone()
            if not exists_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"존재하지 않는 사용자 ID입니다: {user_id}",
                )

            # 2) 기존 설정 조회
            cur.execute(
                """
                SELECT sleep_time, wake_up_time, is_awake_check_enabled, is_alarm_enabled
                FROM user_setting
                WHERE user_id = %s
                LIMIT 1
                """,
                (user_id,),
            )
            row = cur.fetchone()

            sleep_time_str = payload.sleep_time.strftime("%H:%M:%S")
            wake_up_time_str = payload.wake_up_time.strftime("%H:%M:%S")

            if not row:
                # 아직 설정이 없으면 기본 토글값(끄기)으로 INSERT
                insert_sql = """
                    INSERT INTO user_setting (
                        user_id,
                        sleep_time,
                        wake_up_time,
                        is_awake_check_enabled,
                        is_alarm_enabled,
                        created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, NOW()
                    )
                """
                cur.execute(
                    insert_sql,
                    (
                        user_id,
                        sleep_time_str,
                        wake_up_time_str,
                        0,  # is_awake_check_enabled 기본 0
                        0,  # is_alarm_enabled 기본 0
                    ),
                )
                logger.info(
                    "[api_update_sleep_schedule] inserted user_setting for user_id=%s",
                    user_id,
                )
            else:
                # 기존 row가 있으면 취침/기상 시간만 업데이트
                update_sql = """
                    UPDATE user_setting
                    SET
                        sleep_time = %s,
                        wake_up_time = %s
                    WHERE user_id = %s
                """
                cur.execute(
                    update_sql,
                    (
                        sleep_time_str,
                        wake_up_time_str,
                        user_id,
                    ),
                )
                logger.info(
                    "[api_update_sleep_schedule] updated sleep_time/wake_up_time for user_id=%s",
                    user_id,
                )

        conn.commit()

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.error("[api_update_sleep_schedule] error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"수면 패턴 업데이트 중 오류가 발생했습니다: {e}",
        )
    finally:
        conn.close()

    # 업데이트 후 최신 설정 반환
    return await api_get_user_settings(user_id)



# @app.put("/api/users/{user_id}/settings")
# async def api_update_user_settings(user_id: int, payload: UserSettingToggleRequest):
#     """
#     Settings 화면에서 각성 알림 / 알람 on/off만 수정할 때 사용하는 API.
#     - sleep_time, wake_up_time은 건드리지 않음 (온보딩 /api/user-setting에서 관리)
#     - User_Setting row가 없으면 기본 시간(23:00/07:00) + 토글 값으로 생성
#     """
#     conn = get_db_connection()
#     try:
#         with conn.cursor() as cur:
#             # User 존재 여부 확인
#             cur.execute("SELECT 1 FROM User WHERE user_id = %s LIMIT 1", (user_id,))
#             exists_user = cur.fetchone()
#             if not exists_user:
#                 raise HTTPException(
#                     status_code=status.HTTP_400_BAD_REQUEST,
#                     detail=f"존재하지 않는 사용자 ID입니다: {user_id}",
#                 )

#             # User_Setting 존재 여부 확인
#             cur.execute(
#                 "SELECT sleep_time, wake_up_time FROM User_Setting WHERE user_id = %s LIMIT 1",
#                 (user_id,),
#             )
#             row = cur.fetchone()

#             if not row:
#                 # 아직 설정이 없다면 기본 시간으로 INSERT
#                 insert_sql = """
#                     INSERT INTO User_Setting (
#                         user_id,
#                         sleep_time,
#                         wake_up_time,
#                         is_awake_check_enabled,
#                         is_alarm_enabled,
#                         created_at
#                     ) VALUES (
#                         %s, %s, %s, %s, %s, NOW()
#                     )
#                 """
#                 cur.execute(
#                     insert_sql,
#                     (
#                         user_id,
#                         "23:00:00",
#                         "07:00:00",
#                         int(payload.is_awake_check_enabled),
#                         int(payload.is_alarm_enabled),
#                     ),
#                 )
#             else:
#                 # 기존 row가 있으면 토글만 업데이트
#                 update_sql = """
#                     UPDATE User_Setting
#                     SET
#                         is_awake_check_enabled = %s,
#                         is_alarm_enabled = %s
#                     WHERE user_id = %s
#                 """
#                 cur.execute(
#                     update_sql,
#                     (
#                         int(payload.is_awake_check_enabled),
#                         int(payload.is_alarm_enabled),
#                         user_id,
#                     ),
#                 )

#         conn.commit()

#     except HTTPException:
#         conn.rollback()
#         raise
#     except Exception as e:
#         conn.rollback()
#         logger.error("[api_update_user_settings] error: %s", e)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"사용자 설정 업데이트 중 오류가 발생했습니다: {e}",
#         )
#     finally:
#         conn.close()

#     # 업데이트 후 최신 설정 반환
#     return await api_get_user_settings(user_id)







# ============================
# 6. 사용자 설정 API
# ============================

@app.post("/api/user-setting")
async def api_save_user_setting(payload: UserSettingRequest):
    logger.info(
        "[api_save_user_setting] called: user_id=%s, sleep_time=%s, wake_up_time=%s, awake_check=%s, alarm=%s",
        payload.user_id,
        payload.sleep_time,
        payload.wake_up_time,
        payload.is_awake_check_enabled,
        payload.is_alarm_enabled,
    )

    upsert_user_setting(payload)

    return {
        "status": "ok",
        "message": "사용자 설정이 저장되었습니다.",
        "user_id": payload.user_id,
    }

# ============================
# 5. 헬스 체크
# ============================

@app.get("/health")
async def health():
    return {"status": "ok"}

# ============================
# 6. 녹음 파일 저장 폴더
# ============================

os.makedirs("recordings", exist_ok=True)

# # ============================
# # 7. .wav → YAMNet 스펙트로그램 함수
# # ============================

# def wav_to_yamnet_spectrogram(path: str, target_sr: int = 16000) -> np.ndarray:
#     """
#     .wav 파일 경로를 받아서:
#       - 16kHz mono waveform 로딩 (librosa가 자동 리샘플링)
#       - YAMNet으로 spectrogram 추출
#     """
#     logger.info("[yamnet] 1. librosa.load start (wav): %s", path)
#     y, sr = librosa.load(path, sr=target_sr, mono=True)
#     logger.info("[yamnet] 2. librosa.load done: len(y)=%d, sr=%d", len(y), sr)

#     waveform = tf.convert_to_tensor(y, dtype=tf.float32)
#     logger.info("[yamnet] 3. waveform tensor created: shape=%s", waveform.shape)

#     yamnet = get_yamnet_model()
#     logger.info("[yamnet] 4. YAMNet model loaded")

#     scores, embeddings, spectrogram = yamnet(waveform)  # spectrogram: (T, 64)
#     logger.info(
#         "[yamnet] 5. yamnet forward done: spectrogram shape=%s",
#         spectrogram.shape,
#     )

#     spec_np = spectrogram.numpy().astype("float32")
#     logger.info("[yamnet] 6. spectrogram numpy converted: shape=%s", spec_np.shape)

#     return spec_np

# ============================
# [수정] WAV → YAMNet spectrogram (tf.signal.resample 제거 버전)
# ============================
def wav_to_yamnet_spectrogram(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    .wav 파일을 TensorFlow로 읽어와서
    - tf.audio.decode_wav 로 파싱
    - 필요하면 numpy로 16kHz 리샘플
    - YAMNet에 넣어서 spectrogram (3번째 출력)을 numpy로 반환

    반환값: spectrogram.numpy()  # shape: (T, 64) 등
    """
    logger.info("[yamnet] 1. tf.io.read_file start (wav): %s", path)
    audio_binary = tf.io.read_file(path)

    # decode_wav: [-1.0, 1.0] float32, (samples, channels)
    waveform, sample_rate = tf.audio.decode_wav(
        audio_binary,
        desired_channels=1,   # mono 강제
    )  # waveform: (N, 1)
    waveform = tf.squeeze(waveform, axis=-1)  # (N,)
    sample_rate = int(sample_rate.numpy())
    logger.info(
        "[yamnet] 2. tf.audio.decode_wav done: len=%d, sr=%d",
        waveform.shape[0],
        sample_rate,
    )

    # === [핵심 수정] numpy로 16kHz 리샘플 ===
    if sample_rate != target_sr:
        logger.info(
            "[yamnet] 3. numpy resample from %d Hz to %d Hz",
            sample_rate,
            target_sr,
        )
        # TF tensor -> numpy
        wave_np = waveform.numpy().astype("float32")

        # 전체 길이(초)
        duration = wave_np.shape[0] / float(sample_rate)
        # 리샘플 후 길이
        new_length = int(round(duration * target_sr))

        # 원래/새 시간축 생성
        old_times = np.linspace(0.0, duration, num=wave_np.shape[0], endpoint=False)
        new_times = np.linspace(0.0, duration, num=new_length, endpoint=False)

        # 선형 보간으로 리샘플
        wave_np = np.interp(new_times, old_times, wave_np).astype("float32")

        # 다시 TF tensor로 변환
        waveform = tf.convert_to_tensor(wave_np, dtype=tf.float32)
        logger.info(
            "[yamnet] 4. resample done: new_len=%d",
            new_length,
        )
    else:
        # 샘플레이트가 이미 16kHz인 경우
        waveform = tf.cast(waveform, tf.float32)

    # YAMNet 모델 추론
    yamnet = get_yamnet_model()
    logger.info("[yamnet] 5. YAMNet forward start")
    scores, embeddings, spectrogram = yamnet(waveform)
    logger.info(
        "[yamnet] 6. YAMNet forward done: spectrogram shape=%s",
        spectrogram.shape,
    )

    # best_model.keras 에 넘기기 위해 numpy로 변환
    spec_np = spectrogram.numpy().astype("float32")
    return spec_np


# ============================
# 8. 업로드된 10초 청크 처리
# ============================

@app.post("/upload-audio")
async def upload_audio(
    user_id: int = Query(..., description="User ID (쿼리 파라미터 ?user_id=1 형태)"),
    file: UploadFile = File(...),
):
    """
    10초 오디오 청크 업로드 → YAMNet 임베딩(1024) → test.keras 예측 → Sleep_Report 업데이트.

    - user_id: 쿼리 파라미터 (?user_id=1)
    - file   : Form-data로 들어오는 오디오 파일(.webm)
    """
    logger.info("[upload-audio] called (user_id=%s, filename=%s)", user_id, file.filename)

    # 0) user_id 유효성 체크
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            logger.info("[upload-audio] 1. checking user...")
            cur.execute("SELECT 1 FROM user WHERE user_id = %s LIMIT 1", (user_id,))
            exists = cur.fetchone()
            if not exists:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"존재하지 않는 사용자 ID입니다: {user_id}",
                )
    finally:
        conn.close()

    # 1) 파일 저장 (KST 기준 파일명)
    logger.info("[upload-audio] 2. saving file...")
    filename = datetime.now(KST).strftime("%Y%m%d_%H%M%S_%f") + ".wav"
    path = os.path.join("recordings", filename)

    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        try:
            file_size = os.path.getsize(path)
        except OSError:
            file_size = -1
        logger.info(
            "[upload-audio] file saved: %s (size=%d bytes, user_id=%s)",
            path,
            file_size,
            user_id,
        )
    except Exception as e:
        logger.error("[upload-audio] failed to save file: %s (%s)", path, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"녹음 파일 저장에 실패했습니다: {e}",
        )

    # # 2) .webm → YAMNet 임베딩(1024)
    # try:
    #     logger.info("[upload-audio] 3. webm_to_yamnet_embedding start")
    #     embedding_1024 = webm_to_yamnet_embedding(path)  # (1024,)
    #     logger.info("[upload-audio] 4. webm_to_yamnet_embedding done")
    # except Exception as e:
    #     logger.error("[upload-audio] error in webm_to_yamnet_embedding: %s", e)
    #     # 변환 실패 시 파일 삭제
    #     try:
    #         os.remove(path)
    #     except OSError:
    #         pass
    #     raise HTTPException(
    #         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #         detail=f"오디오 처리 중 오류가 발생했습니다: {e}",
    #     )


    # 2) .webm → YAMNet 스펙트로그램으로 변환
    try:
        logger.info("[upload-audio] 3. wav_to_yamnet_spectrogram start")
        spectrogram = wav_to_yamnet_spectrogram(path)  # (T, 64)
        logger.info(
            "[upload-audio] 4. wav_to_yamnet_spectrogram done, shape=%s",
            spectrogram.shape,
        )
    except Exception as e:
        logger.error("[upload-audio] error in wav_to_yamnet_spectrogram: %s", e)
        # 변환 실패 시 파일 삭제
        try:
            os.remove(path)
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"오디오 처리 중 오류가 발생했습니다: {e}",
        )    

    # 3) 모델 예측 (입력: (1, 1024) 가정)
    try:
        logger.info("[upload-audio] 5. model.predict start")
        # x = np.expand_dims(embedding_1024, axis=0)  # (1, 1024)

        # [수정] 시간축(T)을 평균 내서 길이 64 벡터로 변환
        spec_mean = np.mean(spectrogram, axis=0).astype("float32")  # (64,)
        # [수정] 모델이 기대하는 (1, 64, 1) 형태로 reshape
        x = spec_mean.reshape(1, 64, 1)

        preds = model.predict(x)
        logger.info("[upload-audio] 6. model.predict done")
        probs = preds[0]
        label = int(np.argmax(probs))
        max_prob = float(np.max(probs))
        label_name = LABEL_MAP.get(label, "Unknown")
        logger.info(
            "[upload-audio] prediction -> label=%s (%s), max_prob=%.4f",
            label,
            label_name,
            max_prob,
        )
    except Exception as e:
        logger.error("[upload-audio] model.predict error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"모델 예측 중 오류가 발생했습니다: {e}",
        )

    # 4) Sleep_Report 업데이트
    try:
        logger.info("[upload-audio] 7. update_sleep_report start")
        update_sleep_report(user_id=user_id, label=label)
        logger.info("[upload-audio] 8. update_sleep_report done")
    except Exception as e:
        logger.error("[upload-audio] update_sleep_report error: %s", e)
        # DB 에러 시 파일은 남겨두고 에러 반환
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"수면 리포트 갱신 중 오류가 발생했습니다: {e}",
        )

    # 5) 정상(0)인 경우 파일 삭제, 이상(1~4)은 남겨둠(추후 분석/재학습용)
    saved = True
    if label == 0:
        try:
            os.remove(path)
        except OSError:
            pass
        saved = False

    logger.info("[upload-audio] 9. returning response")

    return {
        "status": "ok",
        "user_id": user_id,
        "filename": filename,
        "label": label,
        "label_name": label_name,
        "max_prob": max_prob,
        "saved": saved,
    }

# ============================
# 9. /predict : 리포트 조회
# ============================

@app.get("/predict")
async def predict(user_id: int, date: Optional[str] = None):
    """
    오늘(또는 특정 날짜)의 Sleep_Report 리포트를 조회.

    - user_id: 필수 (쿼리 파라미터)
    - date: 선택 (YYYY-MM-DD 문자열)
      - 없으면 MySQL CURDATE() 기준 오늘 날짜 리포트 조회
    """
    try:
        report = get_sleep_report(user_id=user_id, report_date=date)
    except Exception as e:
        logger.error("[predict] Failed to fetch report: %s", e)
        return {"status": "error", "message": f"Failed to fetch report: {e}"}

    if not report:
        return {
            "status": "not_found",
            "message": "리포트가 없습니다. (해당 날짜에 수면 분석 데이터가 없을 수 있음)",
        }

    return {
        "status": "ok",
        "user_id": report["user_id"],
        "report_date": str(report["report_date"]) if report["report_date"] else None,
        "start_analysis_time": (
            report["start_analysis_time"].isoformat()
            if report["start_analysis_time"] else None
        ),
        "end_analysis_time": (
            report["end_analysis_time"].isoformat()
            if report["end_analysis_time"] else None
        ),
        "sleep_analysis_time": (
            str(report["sleep_analysis_time"])
            if report["sleep_analysis_time"] else None
        ),
        "apnea_level_0": report["apnea_level_0"],
        "apnea_level_1": report["apnea_level_1"],
        "apnea_level_2": report["apnea_level_2"],
        "apnea_level_3": report["apnea_level_3"]
    }
