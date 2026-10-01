-- Label = 1 if the customer placed another delivered order in [:cutoff, :end)
SELECT DISTINCT c.customer_unique_id, 1 AS repeat_purchase
FROM orders ord
JOIN customers c ON ord.customer_id = c.customer_id
WHERE ord.order_status = 'delivered'
  AND ord.order_purchase_timestamp >= :cutoff
  AND ord.order_purchase_timestamp <  :end;
