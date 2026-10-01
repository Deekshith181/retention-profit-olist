-- One row per real customer (customer_unique_id), using ONLY delivered orders before :cutoff.
-- Payments, reviews and items are pre-aggregated per order to avoid row fan-out in the joins.
WITH o AS (
  SELECT c.customer_unique_id AS cid, c.customer_state AS state, ord.order_id,
         ord.order_purchase_timestamp AS ts,
         ord.order_delivered_customer_date AS delivered_at,
         ord.order_estimated_delivery_date AS estimated_at
  FROM orders ord
  JOIN customers c ON ord.customer_id = c.customer_id
  WHERE ord.order_status = 'delivered'
    AND ord.order_purchase_timestamp < :cutoff
),
pay AS (SELECT order_id, SUM(payment_value) AS pay_value, MAX(payment_installments) AS installments
        FROM order_payments GROUP BY order_id),
rev AS (SELECT order_id, AVG(review_score) AS score FROM order_reviews GROUP BY order_id),
itm AS (SELECT order_id, COUNT(*) AS n_items, SUM(freight_value) AS freight, SUM(price) AS price
        FROM order_items GROUP BY order_id)
SELECT o.cid AS customer_unique_id,
       MAX(o.state)                                              AS state,
       COUNT(DISTINCT o.order_id)                                AS n_orders,
       julianday(:cutoff) - julianday(MAX(o.ts))                 AS recency_days,
       julianday(:cutoff) - julianday(MIN(o.ts))                 AS tenure_days,
       COALESCE(SUM(pay.pay_value), 0)                           AS total_spend,
       COALESCE(AVG(pay.pay_value), 0)                           AS avg_order_value,
       COALESCE(AVG(pay.installments), 1)                        AS avg_installments,
       COALESCE(SUM(itm.n_items), 0)                             AS n_items,
       COALESCE(SUM(itm.freight) / NULLIF(SUM(itm.price), 0), 0) AS freight_ratio,
       AVG(rev.score)                                            AS avg_review,
       MIN(rev.score)                                            AS min_review,
       AVG(julianday(o.delivered_at) - julianday(o.estimated_at)) AS avg_delivery_delay_days,
       AVG(CASE WHEN o.delivered_at > o.estimated_at THEN 1.0 ELSE 0.0 END) AS late_share
FROM o
LEFT JOIN pay ON o.order_id = pay.order_id
LEFT JOIN rev ON o.order_id = rev.order_id
LEFT JOIN itm ON o.order_id = itm.order_id
GROUP BY o.cid;
