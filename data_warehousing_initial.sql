-- Step 1: Create Tables

-- Create staging table
CREATE TABLE public.staging (
    invoiceno character varying(20),
    stockcode character varying(20),
    description text,
    quantity integer,
    invoicedate timestamp without time zone,
    unitprice numeric(10,2),
    customerid numeric,
    country character varying(100)
);

-- Create dimcustomer table
CREATE TABLE public.dimcustomer (
    customer_id serial PRIMARY KEY,
    customer_number integer,
    country character varying(100)
);

-- Create dimproduct table
CREATE TABLE public.dimproduct (
    product_id serial PRIMARY KEY,
    stock_code character varying(20),
    description text
);

-- Create dimtime table
CREATE TABLE public.dimtime (
    time_id serial PRIMARY KEY,
    invoice_date timestamp without time zone,
    day integer,
    month integer,
    year integer,
    weekday character varying(20)
);

-- Create factsales table
CREATE TABLE public.factsales (
    sales_id serial PRIMARY KEY,
    invoice_no character varying(20),
    product_id integer REFERENCES public.dimproduct(product_id),
    customer_id integer REFERENCES public.dimcustomer(customer_id),
    time_id integer REFERENCES public.dimtime(time_id),
    quantity integer,
    unit_price numeric(10,2)
);

-- Step 2: Load Data into Staging Table
COPY public.staging (invoiceno, stockcode, description, quantity, invoicedate, unitprice, customerid, country)
FROM 'C:\Users\Alicia\Desktop\IS_107_P_C - Copy\cleaned_online_retail.csv'
DELIMITER ',' 
CSV HEADER;

-- Step 3: Populate Dimension Tables

-- Populate dimcustomer table
INSERT INTO public.dimcustomer (customer_number, country)
SELECT DISTINCT customerid::INTEGER, country
FROM public.staging
WHERE customerid IS NOT NULL;

-- Populate dimproduct table
INSERT INTO public.dimproduct (stock_code, description)
SELECT DISTINCT stockcode, description
FROM public.staging
WHERE stockcode IS NOT NULL AND description IS NOT NULL;

-- Populate dimtime table
INSERT INTO public.dimtime (invoice_date, day, month, year, weekday)
SELECT DISTINCT invoicedate,
       EXTRACT(DAY FROM invoicedate) AS day,
       EXTRACT(MONTH FROM invoicedate) AS month,
       EXTRACT(YEAR FROM invoicedate) AS year,
       TO_CHAR(invoicedate, 'Day') AS weekday
FROM public.staging
WHERE invoicedate IS NOT NULL;

-- Step 4: Populate Fact Table

INSERT INTO public.factsales (invoice_no, product_id, customer_id, time_id, quantity, unit_price)
SELECT st.invoiceno,
       dp.product_id,
       dc.customer_id,
       dt.time_id,
       st.quantity,
       st.unitprice
FROM public.staging st
LEFT JOIN public.dimproduct dp ON st.stockcode = dp.stock_code
LEFT JOIN public.dimcustomer dc ON st.customerid::INTEGER = dc.customer_number
LEFT JOIN public.dimtime dt ON st.invoicedate = dt.invoice_date;