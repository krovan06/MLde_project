create Table if NOT exists test_connection (
  id serial primary KEY,
  message TEXT,
  created_at TIMESTAMP DEFAULT now()
);

INSERT into test_connection (message) values ('postgre работает');