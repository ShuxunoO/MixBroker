"""
@file mine_mixbroker_47d_features.py
@brief MixBroker 论文 47 维混币器特征挖掘与存储
@details
    本脚本依据 MixBroker 论文 (Du et al., IEEE TIFS 2024) 提出的 47 维特征，
    从 one_piece schema 下的 Tornado.Cash 交易数据中计算并存储到 2 张宽表，
    供后续图神经网络训练/AML 分析直接消费。

    特征分组:
        - 42 维节点特征 (tc_node_features_42d): 每个 (pool_name, address) 一行
          * 池子交互频数 (15): pool_X_num / pool_X_num_d / pool_X_num_w + num_all/num_d_all/num_w_all
          * 全部 TC 交易时间 (6): early/late/total_gap/min_gap/max_gap/avg_gap
          * 仅 deposit 时间 (6)
          * 仅 withdraw 时间 (6)
          * 全部 gas price (3)
          * 仅 deposit/withdraw gas price (6)
        - 5 维地址对特征 (tc_address_pair_features_5d): 每个 (pool_name, address_a, address_b) 一行
          * tx_num, tx_value, is_bidirectional
          * tx_num_ab/ba, tx_value_ab/ba (方向拆分)
          * tx_num_ratio_ab/ba, tx_value_ratio_ab/ba (方向 ratio)

    核心流程:
        1. 确保 2 张目标表存在 (DDL)
        2. 每个池子单独 worker, SQL 端 GROUP BY + LAG() window 聚合
        3. Python 端用 execute_values + ON CONFLICT 流式 UPSERT
        4. multiprocessing 4 worker 并行处理 4 个池子

@note 设计原则: 全量 SQL 端聚合, Python 端只做流式 UPSERT
        所有地址统一小写, 字典序约定 address_a < address_b
        所有时间戳基于 TC 交易自身 block_timestamp (Unix 秒), 与论文一致
"""
import os
import sys
import time
import logging
from datetime import datetime
from multiprocessing import get_context

import psycopg2
import psycopg2.extras
from tqdm import tqdm

# 把项目根目录加入 path, 以便 import util/config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import db_tools, fio
from config import config

# ============================================================
# 常量配置
# ============================================================
## @brief 4 个混币池子, 按 denomination 升序
POOLS = ["0_1ETH", "1ETH", "10ETH", "100ETH"]
## @brief pool_name -> 源表表名片段 (去下划线)
def _suffix(p: str) -> str:
    return p.lower()
## @brief 池子 deposit/withdraw/trace 源表名映射
DEPOSIT_TBL = {p: f"one_piece.tornadocash_{_suffix(p)}_deposit_transfers" for p in POOLS}
WITHDRAW_TBL = {p: f"one_piece.tornadocash_{_suffix(p)}_withdraw_transfers" for p in POOLS}
TRACE_TBL = {p: f"one_piece.tc_{_suffix(p)}_onehop_in_out_traces" for p in POOLS}
## @brief 目标结果表名
NODE_TBL = "one_piece.tc_node_features_42d"
PAIR_TBL = "one_piece.tc_address_pair_features_5d"
## @brief UPSERT 每批行数
BATCH_SIZE = 5000
## @brief 并行 worker 数 (默认 4, 跑全 4 池)
WORKERS = 4
## @brief 日志文件路径
LOG_DIR = "/Shuxun/AML_for_Blockchain/logs"
LOG_FILE = os.path.join(LOG_DIR, "mine_mixbroker_47d_features.log")

# ============================================================
# 日志初始化
# ============================================================
def _init_logging():
    """
    @brief 初始化文件 + 控制台日志。
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

# ============================================================
# DDL
# ============================================================
NODE_TBL_DDL = f"""
CREATE TABLE IF NOT EXISTS {NODE_TBL} (
    id                  BIGSERIAL PRIMARY KEY,
    pool_name           VARCHAR(20) NOT NULL,
    address             VARCHAR(42) NOT NULL,
    address_role        TEXT[]      NOT NULL DEFAULT '{{}}',

    -- A. 池子交互频数 (15)
    pool_0_1_num        INTEGER     NOT NULL DEFAULT 0,
    pool_1_num          INTEGER     NOT NULL DEFAULT 0,
    pool_10_num         INTEGER     NOT NULL DEFAULT 0,
    pool_100_num        INTEGER     NOT NULL DEFAULT 0,
    num_all             INTEGER     NOT NULL DEFAULT 0,

    pool_0_1_num_d      INTEGER     NOT NULL DEFAULT 0,
    pool_0_1_num_w      INTEGER     NOT NULL DEFAULT 0,
    pool_1_num_d        INTEGER     NOT NULL DEFAULT 0,
    pool_1_num_w        INTEGER     NOT NULL DEFAULT 0,
    pool_10_num_d       INTEGER     NOT NULL DEFAULT 0,
    pool_10_num_w       INTEGER     NOT NULL DEFAULT 0,
    pool_100_num_d      INTEGER     NOT NULL DEFAULT 0,
    pool_100_num_w      INTEGER     NOT NULL DEFAULT 0,
    num_d_all           INTEGER     NOT NULL DEFAULT 0,
    num_w_all           INTEGER     NOT NULL DEFAULT 0,

    -- B. 全部 TC 交易时间 (6)
    early_time          BIGINT      NOT NULL DEFAULT 0,
    late_time           BIGINT      NOT NULL DEFAULT 0,
    total_time_gap      BIGINT      NOT NULL DEFAULT 0,
    min_time_gap        BIGINT      NOT NULL DEFAULT 0,
    max_time_gap        BIGINT      NOT NULL DEFAULT 0,
    avg_time_gap        BIGINT      NOT NULL DEFAULT 0,

    -- C. 仅 deposit 时间 (6)
    early_time_d        BIGINT      NOT NULL DEFAULT 0,
    late_time_d         BIGINT      NOT NULL DEFAULT 0,
    total_time_gap_d    BIGINT      NOT NULL DEFAULT 0,
    min_time_gap_d      BIGINT      NOT NULL DEFAULT 0,
    max_time_gap_d      BIGINT      NOT NULL DEFAULT 0,
    avg_time_gap_d      BIGINT      NOT NULL DEFAULT 0,

    -- D. 仅 withdraw 时间 (6)
    early_time_w        BIGINT      NOT NULL DEFAULT 0,
    late_time_w         BIGINT      NOT NULL DEFAULT 0,
    total_time_gap_w    BIGINT      NOT NULL DEFAULT 0,
    min_time_gap_w      BIGINT      NOT NULL DEFAULT 0,
    max_time_gap_w      BIGINT      NOT NULL DEFAULT 0,
    avg_time_gap_w      BIGINT      NOT NULL DEFAULT 0,

    -- E. 全部 gas price (3)
    min_gasprice_all    NUMERIC(30,0) NOT NULL DEFAULT 0,
    max_gasprice_all    NUMERIC(30,0) NOT NULL DEFAULT 0,
    avg_gasprice_all    NUMERIC(30,0) NOT NULL DEFAULT 0,

    -- F. 仅 deposit/withdraw gas price (6)
    min_gasprice_d      NUMERIC(30,0) NOT NULL DEFAULT 0,
    max_gasprice_d      NUMERIC(30,0) NOT NULL DEFAULT 0,
    avg_gasprice_d      NUMERIC(30,0) NOT NULL DEFAULT 0,
    min_gasprice_w      NUMERIC(30,0) NOT NULL DEFAULT 0,
    max_gasprice_w      NUMERIC(30,0) NOT NULL DEFAULT 0,
    avg_gasprice_w      NUMERIC(30,0) NOT NULL DEFAULT 0,

    -- 冗余诊断列
    tx_count_evidence   INTEGER     NOT NULL DEFAULT 0,

    created_at          TIMESTAMP   NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP   NOT NULL DEFAULT NOW(),

    CONSTRAINT tc_node_features_42d_pk PRIMARY KEY (pool_name, address)
);
"""

NODE_TBL_INDEX_DDL = [
    f"CREATE INDEX IF NOT EXISTS idx_tc_node_features_42d_addr ON {NODE_TBL}(address);",
    f"CREATE INDEX IF NOT EXISTS idx_tc_node_features_42d_pool_role ON {NODE_TBL}(pool_name, address_role);",
]

PAIR_TBL_DDL = f"""
CREATE TABLE IF NOT EXISTS {PAIR_TBL} (
    id                  BIGSERIAL PRIMARY KEY,
    pool_name           VARCHAR(20) NOT NULL,
    address_a           VARCHAR(42) NOT NULL,
    address_b           VARCHAR(42) NOT NULL,

    -- 43 维: 总笔数
    tx_num              INTEGER     NOT NULL DEFAULT 0,
    -- 方向拆分
    tx_num_ab           INTEGER     NOT NULL DEFAULT 0,
    tx_num_ba           INTEGER     NOT NULL DEFAULT 0,

    -- 45 维: 总金额 (ETH)
    tx_value            NUMERIC(38,18) NOT NULL DEFAULT 0,
    tx_value_ab         NUMERIC(38,18) NOT NULL DEFAULT 0,
    tx_value_ba         NUMERIC(38,18) NOT NULL DEFAULT 0,

    -- 44 维: 方向 ratio
    tx_num_ratio_ab     NUMERIC(10,6)  NOT NULL DEFAULT 0,
    tx_num_ratio_ba     NUMERIC(10,6)  NOT NULL DEFAULT 0,
    -- 46 维: 方向 ratio
    tx_value_ratio_ab   NUMERIC(10,6)  NOT NULL DEFAULT 0,
    tx_value_ratio_ba   NUMERIC(10,6)  NOT NULL DEFAULT 0,

    -- 47 维: 双向
    is_bidirectional    BOOLEAN      NOT NULL DEFAULT FALSE,

    -- 分母冗余 (避免 ratio 计算再 join)
    a_out_count         INTEGER      NOT NULL DEFAULT 0,
    a_out_value         NUMERIC(38,18) NOT NULL DEFAULT 0,
    b_out_count         INTEGER      NOT NULL DEFAULT 0,
    b_out_value         NUMERIC(38,18) NOT NULL DEFAULT 0,

    -- 角色冗余
    a_role              TEXT[]       NOT NULL DEFAULT '{{}}',
    b_role              TEXT[]       NOT NULL DEFAULT '{{}}',

    created_at          TIMESTAMP    NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP    NOT NULL DEFAULT NOW(),

    CONSTRAINT tc_address_pair_features_5d_pk PRIMARY KEY (pool_name, address_a, address_b),
    CONSTRAINT tc_address_pair_features_5d_ab_order CHECK (address_a < address_b)
);
"""

PAIR_TBL_INDEX_DDL = [
    f"CREATE INDEX IF NOT EXISTS idx_tc_address_pair_features_5d_a ON {PAIR_TBL}(pool_name, address_a);",
    f"CREATE INDEX IF NOT EXISTS idx_tc_address_pair_features_5d_b ON {PAIR_TBL}(pool_name, address_b);",
    f"CREATE INDEX IF NOT EXISTS idx_tc_address_pair_features_5d_bidir ON {PAIR_TBL}(pool_name, is_bidirectional);",
]

def ensure_node_table(conn):
    """
    @brief 确保节点特征表 + 索引存在。
    @param conn psycopg2 连接对象。
    """
    cur = conn.cursor()
    try:
        cur.execute(NODE_TBL_DDL)
        for ddl in NODE_TBL_INDEX_DDL:
            cur.execute(ddl)
        conn.commit()
        logging.info(f"[DDL] {NODE_TBL} 确保存在")
    except Exception as e:
        conn.rollback()
        logging.error(f"[DDL] 创建 {NODE_TBL} 失败: {e}")
        raise
    finally:
        cur.close()

def ensure_pair_table(conn):
    """
    @brief 确保地址对特征表 + 索引存在。
    @param conn psycopg2 连接对象。
    """
    cur = conn.cursor()
    try:
        cur.execute(PAIR_TBL_DDL)
        for ddl in PAIR_TBL_INDEX_DDL:
            cur.execute(ddl)
        conn.commit()
        logging.info(f"[DDL] {PAIR_TBL} 确保存在")
    except Exception as e:
        conn.rollback()
        logging.error(f"[DDL] 创建 {PAIR_TBL} 失败: {e}")
        raise
    finally:
        cur.close()

# ============================================================
# SQL 模板构造
# ============================================================
def _src_union_sql(pool: str) -> str:
    """
    @brief 构造某池子 8 张 deposit/withdraw UNION 后的 src 子查询 SQL。
    @details
        - deposit 行: address=deposit_address, method='deposit'
        - withdraw 行: address=withdrawal_recipient, method='withdraw'
        - withdraw_nrc 行: address=withdrawal_initiator, method='withdraw_nrc'
          仅在 initiator_type='none_relayer' 时计入
    @param pool 池子名 (e.g. '100ETH')
    @return src CTE 的 SQL 片段
    """
    d_tbl = DEPOSIT_TBL[pool]
    w_tbl = WITHDRAW_TBL[pool]
    return f"""
        SELECT '{pool}'::varchar(20)  AS pool_name,
               lower(d.deposit_address) AS address,
               'deposit'::varchar(10)   AS method,
               EXTRACT(EPOCH FROM d.block_timestamp)::bigint AS unix_ts,
               COALESCE(d.gas_price, 0) AS gas_price
        FROM {d_tbl} d
        WHERE d.deposit_address IS NOT NULL
          AND d.block_timestamp IS NOT NULL
          AND d.pool_name = '{pool}'
        UNION ALL
        SELECT '{pool}'::varchar(20),
               lower(w.withdrawal_recipient),
               'withdraw'::varchar(10),
               EXTRACT(EPOCH FROM w.block_timestamp)::bigint,
               COALESCE(w.gas_price, 0)
        FROM {w_tbl} w
        WHERE w.withdrawal_recipient IS NOT NULL
          AND w.block_timestamp IS NOT NULL
          AND w.pool_name = '{pool}'
        UNION ALL
        SELECT '{pool}'::varchar(20),
               lower(w.withdrawal_initiator),
               'withdraw_nrc'::varchar(12),
               EXTRACT(EPOCH FROM w.block_timestamp)::bigint,
               COALESCE(w.gas_price, 0)
        FROM {w_tbl} w
        WHERE w.withdrawal_initiator IS NOT NULL
          AND w.block_timestamp IS NOT NULL
          AND w.pool_name = '{pool}'
          AND w.initiator_type = 'none_relayer'
    """

def _all_pool_counts_union_sql() -> str:
    """
    @brief 构造 4 池子 × 2 表的 (address, pool_tag, method) UNION。
    @return UNION SQL 片段
    """
    parts = []
    for p in POOLS:
        d_tbl = DEPOSIT_TBL[p]
        w_tbl = WITHDRAW_TBL[p]
        # pool_tag 转换: 0_1ETH -> 0_1
        tag = "0_1" if p == "0_1ETH" else p.replace("ETH", "")
        parts.append(f"""
            SELECT lower(d.deposit_address) AS address, '{tag}' AS pool_tag, 'deposit' AS method
            FROM {d_tbl} d
            WHERE d.deposit_address IS NOT NULL AND d.pool_name = '{p}'
            UNION ALL
            SELECT lower(w.withdrawal_recipient), '{tag}', 'withdraw'
            FROM {w_tbl} w
            WHERE w.withdrawal_recipient IS NOT NULL AND w.pool_name = '{p}'
            UNION ALL
            SELECT lower(w.withdrawal_initiator), '{tag}', 'withdraw_nrc'
            FROM {w_tbl} w
            WHERE w.withdrawal_initiator IS NOT NULL
              AND w.pool_name = '{p}'
              AND w.initiator_type = 'none_relayer'
        """)
    return "\nUNION ALL\n".join(parts)

def _address_role_array_sql(addr_expr: str, pool: str) -> str:
    """
    @brief 构造 address_role TEXT[] 数组 SQL 表达式。
    @param addr_expr 地址列表达式 (如 'u.address' 或 'pa.address_a')
    @param pool 池子名
    @return SQL 表达式
    """
    d_tbl = DEPOSIT_TBL[pool]
    w_tbl = WITHDRAW_TBL[pool]
    return f"""
        (SELECT COALESCE(array_agg(role) FILTER (WHERE role IS NOT NULL), '{{}}'::text[])
         FROM unnest(ARRAY[
            CASE WHEN EXISTS (SELECT 1 FROM {d_tbl} d
                              WHERE d.pool_name='{pool}' AND lower(d.deposit_address)={addr_expr})
                 THEN 'deposit' END,
            CASE WHEN EXISTS (SELECT 1 FROM {w_tbl} w
                              WHERE w.pool_name='{pool}' AND lower(w.withdrawal_recipient)={addr_expr})
                 THEN 'withdraw' END,
            CASE WHEN EXISTS (SELECT 1 FROM {w_tbl} w
                              WHERE w.pool_name='{pool}' AND lower(w.withdrawal_initiator)={addr_expr}
                                AND w.initiator_type='none_relayer')
                 THEN 'none_relayer_initiator' END
         ]) AS role)
    """

def build_node_sql(pool: str) -> str:
    """
    @brief 构造某池子的 42 维节点特征大 SQL。
    @details
        一次 SQL 跑完: 池子交互频数(15) + 时间(18) + gas(9) + 角色
        返回 1 列 address + 1 列 address_role + 42 数值列 + tx_count_evidence
    @param pool 池子名
    @return 完整 SELECT SQL
    """
    src_sql = _src_union_sql(pool)
    counts_sql = _all_pool_counts_union_sql()
    role_sql = _address_role_array_sql("u.address", pool)

    return f"""
WITH
src AS ({src_sql}),
counts_src AS ({counts_sql}),
counts AS (
    SELECT address,
           SUM((pool_tag='0_1')::int)                              AS pool_0_1_num,
           SUM((pool_tag='1')::int)                                AS pool_1_num,
           SUM((pool_tag='10')::int)                               AS pool_10_num,
           SUM((pool_tag='100')::int)                              AS pool_100_num,
           SUM((pool_tag='0_1' AND method='deposit')::int)         AS pool_0_1_num_d,
           SUM((pool_tag='0_1' AND method IN ('withdraw','withdraw_nrc'))::int) AS pool_0_1_num_w,
           SUM((pool_tag='1'   AND method='deposit')::int)         AS pool_1_num_d,
           SUM((pool_tag='1'   AND method IN ('withdraw','withdraw_nrc'))::int) AS pool_1_num_w,
           SUM((pool_tag='10'  AND method='deposit')::int)         AS pool_10_num_d,
           SUM((pool_tag='10'  AND method IN ('withdraw','withdraw_nrc'))::int) AS pool_10_num_w,
           SUM((pool_tag='100' AND method='deposit')::int)         AS pool_100_num_d,
           SUM((pool_tag='100' AND method IN ('withdraw','withdraw_nrc'))::int) AS pool_100_num_w,
           SUM((method='deposit')::int)                            AS num_d_all,
           SUM((method IN ('withdraw','withdraw_nrc'))::int)       AS num_w_all,
           COUNT(*)                                                AS num_all
    FROM counts_src
    GROUP BY address
),
ts_all AS (
    SELECT address, unix_ts,
           LAG(unix_ts) OVER (PARTITION BY address ORDER BY unix_ts) AS prev_ts
    FROM src
),
time_all AS (
    SELECT address,
           MIN(unix_ts) AS early_time,
           MAX(unix_ts) AS late_time,
           COALESCE(MAX(unix_ts) - MIN(unix_ts), 0) AS total_time_gap,
           COALESCE(MIN(unix_ts - prev_ts) FILTER (WHERE prev_ts IS NOT NULL), 0) AS min_time_gap,
           COALESCE(MAX(unix_ts - prev_ts) FILTER (WHERE prev_ts IS NOT NULL), 0) AS max_time_gap,
           COALESCE(AVG(unix_ts - prev_ts) FILTER (WHERE prev_ts IS NOT NULL)::bigint, 0) AS avg_time_gap
    FROM ts_all GROUP BY address
),
ts_d AS (
    SELECT address, unix_ts,
           LAG(unix_ts) OVER (PARTITION BY address ORDER BY unix_ts) AS prev_ts
    FROM src WHERE method='deposit'
),
time_d AS (
    SELECT address,
           MIN(unix_ts) AS early_time_d,
           MAX(unix_ts) AS late_time_d,
           COALESCE(MAX(unix_ts) - MIN(unix_ts), 0) AS total_time_gap_d,
           COALESCE(MIN(unix_ts - prev_ts) FILTER (WHERE prev_ts IS NOT NULL), 0) AS min_time_gap_d,
           COALESCE(MAX(unix_ts - prev_ts) FILTER (WHERE prev_ts IS NOT NULL), 0) AS max_time_gap_d,
           COALESCE(AVG(unix_ts - prev_ts) FILTER (WHERE prev_ts IS NOT NULL)::bigint, 0) AS avg_time_gap_d
    FROM ts_d GROUP BY address
),
ts_w AS (
    SELECT address, unix_ts,
           LAG(unix_ts) OVER (PARTITION BY address ORDER BY unix_ts) AS prev_ts
    FROM src WHERE method IN ('withdraw','withdraw_nrc')
),
time_w AS (
    SELECT address,
           MIN(unix_ts) AS early_time_w,
           MAX(unix_ts) AS late_time_w,
           COALESCE(MAX(unix_ts) - MIN(unix_ts), 0) AS total_time_gap_w,
           COALESCE(MIN(unix_ts - prev_ts) FILTER (WHERE prev_ts IS NOT NULL), 0) AS min_time_gap_w,
           COALESCE(MAX(unix_ts - prev_ts) FILTER (WHERE prev_ts IS NOT NULL), 0) AS max_time_gap_w,
           COALESCE(AVG(unix_ts - prev_ts) FILTER (WHERE prev_ts IS NOT NULL)::bigint, 0) AS avg_time_gap_w
    FROM ts_w GROUP BY address
),
gas_all AS (
    SELECT address,
           MIN(gas_price) AS min_gasprice_all,
           MAX(gas_price) AS max_gasprice_all,
           COALESCE(AVG(gas_price), 0)::numeric(30,0) AS avg_gasprice_all
    FROM src GROUP BY address
),
gas_d AS (
    SELECT address,
           MIN(gas_price) AS min_gasprice_d,
           MAX(gas_price) AS max_gasprice_d,
           COALESCE(AVG(gas_price), 0)::numeric(30,0) AS avg_gasprice_d
    FROM src WHERE method='deposit' GROUP BY address
),
gas_w AS (
    SELECT address,
           MIN(gas_price) AS min_gasprice_w,
           MAX(gas_price) AS max_gasprice_w,
           COALESCE(AVG(gas_price), 0)::numeric(30,0) AS avg_gasprice_w
    FROM src WHERE method IN ('withdraw','withdraw_nrc') GROUP BY address
),
addr_universe AS (
    SELECT address FROM counts
    UNION
    SELECT address FROM time_all
    UNION
    SELECT address FROM time_d
    UNION
    SELECT address FROM time_w
    UNION
    SELECT address FROM gas_all
    UNION
    SELECT address FROM gas_d
    UNION
    SELECT address FROM gas_w
)
SELECT
    '{pool}'::varchar(20)             AS pool_name,
    u.address,
    {role_sql}                        AS address_role,

    COALESCE(c.pool_0_1_num, 0),
    COALESCE(c.pool_1_num, 0),
    COALESCE(c.pool_10_num, 0),
    COALESCE(c.pool_100_num, 0),
    COALESCE(c.num_all, 0),

    COALESCE(c.pool_0_1_num_d, 0),
    COALESCE(c.pool_0_1_num_w, 0),
    COALESCE(c.pool_1_num_d, 0),
    COALESCE(c.pool_1_num_w, 0),
    COALESCE(c.pool_10_num_d, 0),
    COALESCE(c.pool_10_num_w, 0),
    COALESCE(c.pool_100_num_d, 0),
    COALESCE(c.pool_100_num_w, 0),
    COALESCE(c.num_d_all, 0),
    COALESCE(c.num_w_all, 0),

    COALESCE(ta.early_time, 0),       COALESCE(ta.late_time, 0),
    COALESCE(ta.total_time_gap, 0),   COALESCE(ta.min_time_gap, 0),
    COALESCE(ta.max_time_gap, 0),     COALESCE(ta.avg_time_gap, 0),

    COALESCE(td.early_time_d, 0),     COALESCE(td.late_time_d, 0),
    COALESCE(td.total_time_gap_d, 0), COALESCE(td.min_time_gap_d, 0),
    COALESCE(td.max_time_gap_d, 0),   COALESCE(td.avg_time_gap_d, 0),

    COALESCE(tw.early_time_w, 0),     COALESCE(tw.late_time_w, 0),
    COALESCE(tw.total_time_gap_w, 0), COALESCE(tw.min_time_gap_w, 0),
    COALESCE(tw.max_time_gap_w, 0),   COALESCE(tw.avg_time_gap_w, 0),

    COALESCE(ga.min_gasprice_all, 0), COALESCE(ga.max_gasprice_all, 0), COALESCE(ga.avg_gasprice_all, 0),
    COALESCE(gd.min_gasprice_d, 0),   COALESCE(gd.max_gasprice_d, 0),   COALESCE(gd.avg_gasprice_d, 0),
    COALESCE(gw.min_gasprice_w, 0),   COALESCE(gw.max_gasprice_w, 0),   COALESCE(gw.avg_gasprice_w, 0),

    COALESCE(c.num_all, 0)            AS tx_count_evidence

FROM addr_universe u
LEFT JOIN counts c   USING (address)
LEFT JOIN time_all ta USING (address)
LEFT JOIN time_d td   USING (address)
LEFT JOIN time_w tw   USING (address)
LEFT JOIN gas_all ga  USING (address)
LEFT JOIN gas_d gd    USING (address)
LEFT JOIN gas_w gw    USING (address);
"""

def build_pair_sql(pool: str) -> str:
    """
    @brief 构造某池子的 5 维地址对特征 SQL。
    @details
        - seeds: 池子所有 seed_address 去重
        - cand: from 和 to 都 ∈ seeds 的 trace 行 (排除 from=to, value<=0)
        - ordered: LEAST/GREATEST 字典序化 pair
        - pair_agg: GROUP BY pair 算 7 个聚合
        - denom: 每地址出账笔数/金额
        - 最终 SELECT 计算 4 ratio + is_bidirectional + 角色
    @param pool 池子名
    @return 完整 SELECT SQL
    """
    t_tbl = TRACE_TBL[pool]
    role_a_sql = _address_role_array_sql("pa.address_a", pool)
    role_b_sql = _address_role_array_sql("pa.address_b", pool)

    return f"""
WITH
params AS (SELECT '{pool}'::varchar(20) AS p),
seeds AS (
    SELECT DISTINCT t.seed_address
    FROM {t_tbl} t
    WHERE t.pool_name = (SELECT p FROM params)
      AND t.seed_address IS NOT NULL
),
cand AS (
    SELECT t.tx_hash, t.from_address, t.to_address, t.value
    FROM {t_tbl} t
    JOIN seeds s1 ON s1.seed_address = t.from_address
    JOIN seeds s2 ON s2.seed_address = t.to_address
    WHERE t.pool_name = (SELECT p FROM params)
      AND t.from_address IS NOT NULL
      AND t.to_address IS NOT NULL
      AND t.from_address <> t.to_address
      AND t.value > 0
),
ordered AS (
    SELECT LEAST(from_address, to_address)  AS address_a,
           GREATEST(from_address, to_address) AS address_b,
           from_address, to_address, value
    FROM cand
),
pair_agg AS (
    SELECT address_a, address_b,
           COUNT(*)                                                          AS tx_num,
           SUM(CASE WHEN from_address = address_a THEN 1 ELSE 0 END)         AS tx_num_ab,
           SUM(CASE WHEN from_address = address_b THEN 1 ELSE 0 END)         AS tx_num_ba,
           SUM(value)                                                        AS tx_value,
           SUM(CASE WHEN from_address = address_a THEN value ELSE 0 END)     AS tx_value_ab,
           SUM(CASE WHEN from_address = address_b THEN value ELSE 0 END)     AS tx_value_ba
    FROM ordered
    GROUP BY address_a, address_b
),
denom AS (
    SELECT t.from_address AS address,
           COUNT(*)                        AS out_count,
           SUM(t.value)                    AS out_value
    FROM {t_tbl} t
    WHERE t.pool_name = (SELECT p FROM params)
      AND t.from_address IS NOT NULL
      AND t.from_address <> t.to_address
      AND t.value > 0
    GROUP BY t.from_address
)
SELECT
    (SELECT p FROM params)                                            AS pool_name,
    pa.address_a,
    pa.address_b,

    pa.tx_num,
    pa.tx_num_ab,
    pa.tx_num_ba,

    pa.tx_value,
    pa.tx_value_ab,
    pa.tx_value_ba,

    CASE WHEN COALESCE(d_a.out_count, 0) > 0
         THEN round((pa.tx_num_ab::numeric / d_a.out_count)::numeric, 6)
         ELSE 0 END                                                   AS tx_num_ratio_ab,
    CASE WHEN COALESCE(d_b.out_count, 0) > 0
         THEN round((pa.tx_num_ba::numeric / d_b.out_count)::numeric, 6)
         ELSE 0 END                                                   AS tx_num_ratio_ba,
    CASE WHEN COALESCE(d_a.out_value, 0) > 0
         THEN round((pa.tx_value_ab::numeric / d_a.out_value)::numeric, 6)
         ELSE 0 END                                                   AS tx_value_ratio_ab,
    CASE WHEN COALESCE(d_b.out_value, 0) > 0
         THEN round((pa.tx_value_ba::numeric / d_b.out_value)::numeric, 6)
         ELSE 0 END                                                   AS tx_value_ratio_ba,

    (pa.tx_num_ab > 0 AND pa.tx_num_ba > 0)                           AS is_bidirectional,

    COALESCE(d_a.out_count, 0)                                        AS a_out_count,
    COALESCE(d_a.out_value, 0)                                        AS a_out_value,
    COALESCE(d_b.out_count, 0)                                        AS b_out_count,
    COALESCE(d_b.out_value, 0)                                        AS b_out_value,

    {role_a_sql}                                                      AS a_role,
    {role_b_sql}                                                      AS b_role

FROM pair_agg pa
LEFT JOIN denom d_a ON d_a.address = pa.address_a
LEFT JOIN denom d_b ON d_b.address = pa.address_b;
"""

# ============================================================
# 通用 UPSERT
# ============================================================
def upsert_rows(conn, table: str, cols: list, rows: list, conflict_keys: tuple) -> int:
    """
    @brief 通用流式 UPSERT (execute_values + ON CONFLICT DO UPDATE)。
    @details
        - 自动跳过 created_at 列 (不更新)
        - 自动跳过 conflict_keys 列 (用于 DO UPDATE 排除主键)
        - 分批 BATCH_SIZE commit
    @param conn 数据库连接
    @param table 目标表名 (含 schema)
    @param cols 列名列表 (顺序与 rows 对齐)
    @param rows 数据行列表
    @param conflict_keys 主键列名 tuple
    @return 写入总行数
    """
    if not rows:
        return 0

    # 不更新的列: 主键 + created_at
    skip_update = set(conflict_keys) | {"id", "created_at"}
    update_cols = [c for c in cols if c not in skip_update]
    if "updated_at" in cols and "updated_at" not in update_cols:
        update_cols.append("updated_at")

    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    insert_cols = ", ".join(cols)
    sql = f"""
        INSERT INTO {table} ({insert_cols}) VALUES %s
        ON CONFLICT ({', '.join(conflict_keys)}) DO UPDATE SET {set_clause};
    """

    total = 0
    cur = conn.cursor()
    try:
        for i in tqdm(range(0, len(rows), BATCH_SIZE),
                      desc=f"UPSERT {table.split('.')[-1]}",
                      unit="batch"):
            batch = rows[i:i + BATCH_SIZE]
            psycopg2.extras.execute_values(cur, sql, batch, page_size=BATCH_SIZE)
            conn.commit()
            total += len(batch)
    except Exception as e:
        conn.rollback()
        logging.error(f"[UPSERT] {table} 失败: {e}")
        raise
    finally:
        cur.close()
    return total

# ============================================================
# 单池子计算
# ============================================================
def compute_node_features_for_pool(conn, pool: str) -> int:
    """
    @brief 计算并写入某池子的 42 维节点特征。
    @param conn 数据库连接
    @param pool 池子名
    @return 写入行数
    """
    t0 = time.time()
    sql = build_node_sql(pool)
    cur = conn.cursor()
    try:
        logging.info(f"[{pool}] 42 维 SQL 开始执行...")
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    except Exception as e:
        conn.rollback()
        logging.error(f"[{pool}] 42 维 SQL 失败: {e}")
        raise
    finally:
        cur.close()

    logging.info(f"[{pool}] 42 维 SQL 完成, {len(rows)} 行, 耗时 {time.time()-t0:.1f}s")
    n = upsert_rows(conn, NODE_TBL, cols, rows, conflict_keys=("pool_name", "address"))
    logging.info(f"[{pool}] 42 维 UPSERT 完成: {n} 行")
    return n

def compute_pair_features_for_pool(conn, pool: str) -> int:
    """
    @brief 计算并写入某池子的 5 维地址对特征。
    @param conn 数据库连接
    @param pool 池子名
    @return 写入行数
    """
    t0 = time.time()
    sql = build_pair_sql(pool)
    cur = conn.cursor()
    try:
        logging.info(f"[{pool}] 5 维 pair SQL 开始执行...")
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    except Exception as e:
        conn.rollback()
        logging.error(f"[{pool}] 5 维 pair SQL 失败: {e}")
        raise
    finally:
        cur.close()

    logging.info(f"[{pool}] 5 维 pair SQL 完成, {len(rows)} 行, 耗时 {time.time()-t0:.1f}s")
    if not rows:
        return 0
    n = upsert_rows(conn, PAIR_TBL, cols, rows,
                    conflict_keys=("pool_name", "address_a", "address_b"))
    logging.info(f"[{pool}] 5 维 pair UPSERT 完成: {n} 行")
    return n

# ============================================================
# 单池子 worker (multiprocessing)
# ============================================================
def process_one_pool(pool_name: str) -> dict:
    """
    @brief 单池子 worker 入口。每个 worker 独立 connect_db。
    @param pool_name 池子名
    @return {"pool": ..., "node_rows": ..., "pair_rows": ..., "elapsed": ..., "status": ...}
    """
    t0 = time.time()
    result = {"pool": pool_name, "node_rows": 0, "pair_rows": 0, "elapsed": 0.0, "status": "ok"}
    conn = db_tools.connect_db()
    if not conn:
        result["status"] = "connect_failed"
        logging.error(f"[{pool_name}] 数据库连接失败")
        return result
    try:
        ensure_node_table(conn)
        ensure_pair_table(conn)
        result["node_rows"] = compute_node_features_for_pool(conn, pool_name)
        result["pair_rows"] = compute_pair_features_for_pool(conn, pool_name)
    except Exception as e:
        result["status"] = f"error: {e}"
        logging.exception(f"[{pool_name}] 处理失败: {e}")
    finally:
        conn.close()
    result["elapsed"] = time.time() - t0
    logging.info(f"[{pool_name}] 完成: nodes={result['node_rows']}, pairs={result['pair_rows']}, "
                 f"耗时={result['elapsed']:.1f}s, status={result['status']}")
    return result

# ============================================================
# 验证
# ============================================================
def verify_table(conn) -> None:
    """
    @brief 跑完后做完整性校验 SQL。
    @param conn 数据库连接
    """
    cur = conn.cursor()
    try:
        logging.info("=" * 60)
        logging.info(" 完整性校验 ")
        logging.info("=" * 60)
        cur.execute(f"""
            SELECT pool_name, COUNT(*) AS node_rows,
                   SUM(num_all) AS sum_num_all,
                   COUNT(*) FILTER (WHERE address_role <> '{{}}'::text[]) AS with_role
            FROM {NODE_TBL}
            GROUP BY pool_name
            ORDER BY pool_name;
        """)
        for r in cur.fetchall():
            logging.info(f"  [NODE] {r[0]}: rows={r[1]}, sum_num_all={r[2]}, with_role={r[3]}")

        cur.execute(f"""
            SELECT pool_name,
                   COUNT(*) AS pair_rows,
                   COUNT(*) FILTER (WHERE is_bidirectional) AS bidir_rows,
                   AVG(tx_num_ratio_ab)::numeric(10,4) AS avg_ratio_ab
            FROM {PAIR_TBL}
            GROUP BY pool_name
            ORDER BY pool_name;
        """)
        for r in cur.fetchall():
            logging.info(f"  [PAIR] {r[0]}: rows={r[1]}, bidirectional={r[2]}, avg_ratio_ab={r[3]}")
    finally:
        cur.close()

# ============================================================
# 入口
# ============================================================
def main(pools: list = None, workers: int = WORKERS):
    """
    @brief 顶层入口: 跑指定池子 (默认 4 池) 的 47 维特征挖掘与存储。
    @details
        - 多进程并行 (spawn context)
        - 跑完做完整性校验
        - 所有步骤写日志
    @param pools 要跑的池子列表, None = 全部 4 池
    @param workers 并行 worker 数 (None 用默认 4)
    """
    _init_logging()
    start = time.time()
    pools = pools or POOLS
    workers = workers or WORKERS

    logging.info("=" * 60)
    logging.info(" MixBroker 47 维特征挖掘与存储 ")
    logging.info(f" Pools: {pools}, Workers: {workers} ")
    logging.info("=" * 60)

    if workers == 1:
        results = [process_one_pool(p) for p in pools]
    else:
        ctx = get_context("spawn")
        with ctx.Pool(min(workers, len(pools))) as pool:
            results = pool.map(process_one_pool, pools)

    # 汇总
    total_nodes = sum(r["node_rows"] for r in results)
    total_pairs = sum(r["pair_rows"] for r in results)
    elapsed = time.time() - start
    logging.info("=" * 60)
    logging.info(f" 全部完成: nodes={total_nodes}, pairs={total_pairs}, "
                 f"elapsed={elapsed:.1f}s")
    for r in results:
        logging.info(f"  - {r['pool']}: nodes={r['node_rows']}, pairs={r['pair_rows']}, "
                     f"elapsed={r['elapsed']:.1f}s, status={r['status']}")
    logging.info("=" * 60)

    # 校验
    try:
        conn = db_tools.connect_db()
        if conn:
            verify_table(conn)
            conn.close()
    except Exception as e:
        logging.error(f"校验失败: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MixBroker 47 维特征挖掘与存储")
    parser.add_argument("--pools", nargs="+", default=None,
                        help="要跑的池子列表, 默认全部 4 池")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"并行 worker 数 (默认 {WORKERS}, 1=单进程)")
    args = parser.parse_args()
    main(pools=args.pools, workers=args.workers)
