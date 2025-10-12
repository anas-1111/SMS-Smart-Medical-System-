USE smart_clinic;


CREATE TABLE specialties (
    specialty_id INT PRIMARY KEY IDENTITY(1,1),
    name NVARCHAR(100) NOT NULL
);

CREATE TABLE doctors (
    doctor_id INT PRIMARY KEY IDENTITY(1,1),
    name NVARCHAR(100) NOT NULL,
    specialty_id INT,
    FOREIGN KEY (specialty_id) REFERENCES specialties(specialty_id)
);

CREATE TABLE patients (
    patient_id INT PRIMARY KEY IDENTITY(1,1),
    username NVARCHAR(50) UNIQUE NOT NULL,
    password NVARCHAR(255) NOT NULL
        CHECK (
            LEN(password) >= 8 
            AND password LIKE '%[0-9]%' 
            AND password LIKE '%[A-Za-z]%'
        ),
    first_name NVARCHAR(50) NOT NULL,
    last_name NVARCHAR(50) NOT NULL,
    gender NVARCHAR(10) CHECK (
        gender IN (N'Male', N'Female', N'male', N'female', N'MALE', N'FEMALE')
    ),
    date_of_birth DATE NOT NULL,
    phone NVARCHAR(20),
    email NVARCHAR(100),
    national_id NVARCHAR(20) UNIQUE,
    address NVARCHAR(200),
    blood_type NVARCHAR(5),
    emergency_contact NVARCHAR(100),
    emergency_phone NVARCHAR(20),
    face_data VARBINARY(MAX) NULL,
    created_at DATETIME DEFAULT GETDATE()
);


CREATE TABLE medical_history (
    history_id INT PRIMARY KEY IDENTITY(1,1),
    patient_id INT NOT NULL,
    chronic_diseases NVARCHAR(500),    
    allergies NVARCHAR(500),          
    current_medications NVARCHAR(500), 
    past_surgeries NVARCHAR(500),      
    family_history NVARCHAR(500),      
    notes NVARCHAR(1000),              
    last_updated DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE schedules (
    schedule_id INT PRIMARY KEY IDENTITY(1,1),
    doctor_id INT,
    available_day NVARCHAR(20),   
    time_from TIME,
    time_to TIME,
    FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
);

CREATE TABLE appointments (
    appointment_id INT PRIMARY KEY IDENTITY(1,1),
    patient_id INT,
    doctor_id INT,
    appointment_day NVARCHAR(20),  
    appointment_time TIME,
    status NVARCHAR(20) DEFAULT 'Booked',
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
);

-- Declare variables
DECLARE @doctor_id INT = 1;
DECLARE @patient_id INT = 2;
DECLARE @appointment_day NVARCHAR(20) = 'Monday';
DECLARE @appointment_time TIME = '10:00';
DECLARE @count INT;

-- Count existing appointments
SELECT @count = COUNT(*)
FROM appointments
WHERE doctor_id = @doctor_id
  AND appointment_day = @appointment_day
  AND DATEPART(HOUR, appointment_time) = DATEPART(HOUR, @appointment_time);

-- Insert if less than 4
IF @count < 4
BEGIN
    INSERT INTO appointments (patient_id, doctor_id, appointment_day, appointment_time, status)
    VALUES (@patient_id, @doctor_id, @appointment_day, @appointment_time, 'Booked'); 
END
ELSE
BEGIN
    PRINT 'Doctor cannot have more than 4 appointments in the same hour.';
END

CREATE TABLE users (
    user_id INT PRIMARY KEY IDENTITY(1,1),
    username NVARCHAR(50) UNIQUE NOT NULL,
    password_hash NVARCHAR(255) NOT NULL,      
    role NVARCHAR(20) CHECK (role IN (N'Patient', N'Doctor', N'Admin')) NOT NULL,
    linked_patient_id INT NULL,
    linked_doctor_id INT NULL,
    face_embedding VARBINARY(MAX) NOT NULL,   
    created_at DATETIME DEFAULT GETDATE(),
    last_login DATETIME NULL,
    FOREIGN KEY (linked_patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (linked_doctor_id) REFERENCES doctors(doctor_id)
);

-- Add new table for storing image analysis records
CREATE TABLE image_analysis (
    analysis_id INT IDENTITY(1,1) PRIMARY KEY,
    patient_id INT NOT NULL,
    analysis_date DATETIME NOT NULL DEFAULT GETDATE(),
    analysis_type NVARCHAR(50) NOT NULL,
    language NVARCHAR(20) NOT NULL,
    result_text NVARCHAR(MAX) NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Add a new column to medical_history table to store image analysis references
IF NOT EXISTS(SELECT * FROM sys.columns 
            WHERE Name = 'image_analyses' AND Object_ID = Object_ID('medical_history'))
BEGIN
    ALTER TABLE medical_history 
    ADD image_analyses NVARCHAR(MAX);
END

-- Update the notes column to NVARCHAR(MAX) if it's not already
IF EXISTS(SELECT * FROM sys.columns 
        WHERE Name = 'notes' AND Object_ID = Object_ID('medical_history')
        AND max_length != -1)
BEGIN
    ALTER TABLE medical_history 
    ALTER COLUMN notes NVARCHAR(MAX);
END

INSERT INTO specialties (name) VALUES
(N'Cardiology'), (N'Dermatology'), (N'Neurology'),
(N'Pediatrics'), (N'Orthopedics'), (N'ENT'),
(N'Ophthalmology'), (N'Psychiatry'), (N'Urology'),
(N'Oncology'), (N'General Surgery'), (N'Radiology');

INSERT INTO doctors (name, specialty_id) VALUES
(N'Dr. Ahmed Hassan', 1),
(N'Dr. Salma Mahmoud', 1),
(N'Dr. Tarek Farouk', 1),

(N'Dr. Mona Ali', 2),
(N'Dr. Nourhan Adel', 2),
(N'Dr. Samir Ibrahim', 2),

(N'Dr. Karim Saad', 3),
(N'Dr. Mai Youssef', 3),
(N'Dr. Omar ElSayed', 3),

(N'Dr. Sara Mostafa', 4),
(N'Dr. Dina Khaled', 4),
(N'Dr. Youssef Amin', 4),

(N'Dr. Omar Fathy', 5),
(N'Dr. Laila Adel', 5),
(N'Dr. Hossam Mohamed', 5),

(N'Dr. Hany Nabil', 6),
(N'Dr. Rana Essam', 6),
(N'Dr. Fady Samir', 6),

(N'Dr. Sherif Adel', 7),
(N'Dr. Marwa Gamal', 7),
(N'Dr. Eman Tarek', 7),

(N'Dr. Mahmoud Reda', 8),
(N'Dr. Hoda Ali', 8),
(N'Dr. Ali Hassan', 8),

(N'Dr. Walid Kamel', 9),
(N'Dr. Heba Saeed', 9),
(N'Dr. Khaled Omar', 9),

(N'Dr. Rasha Ibrahim', 10),
(N'Dr. Amr Fathy', 10),
(N'Dr. Aya Adel', 10),

(N'Dr. Samah Nabil', 11),
(N'Dr. Adel Younes', 11),
(N'Dr. Karim Magdy', 11),

(N'Dr. Tamer Hany', 12),
(N'Dr. Nesma Salah', 12),
(N'Dr. Ihab Lotfy', 12);

INSERT INTO schedules (doctor_id, available_day, time_from, time_to) VALUES
-- Cardiology 
(1, N'Sunday', '09:00', '12:00'), (1, N'Tuesday', '14:00', '17:00'),
(2, N'Monday', '10:00', '13:00'), (2, N'Thursday', '15:00', '18:00'),
(3, N'Wednesday', '11:00', '14:00'), (3, N'Friday', '09:00', '12:00'),

-- Dermatology 
(4, N'Sunday', '13:00', '16:00'), (4, N'Tuesday', '10:00', '13:00'),
(5, N'Monday', '09:00', '12:00'), (5, N'Wednesday', '15:00', '18:00'),
(6, N'Thursday', '11:00', '14:00'), (6, N'Saturday', '10:00', '13:00'),

-- Neurology 
(7, N'Sunday', '09:00', '12:00'), (7, N'Tuesday', '14:00', '17:00'),
(8, N'Monday', '10:00', '13:00'), (8, N'Wednesday', '11:00', '14:00'),
(9, N'Thursday', '15:00', '18:00'), (9, N'Saturday', '09:00', '12:00'),

-- Pediatrics 
(10, N'Sunday', '12:00', '15:00'), (10, N'Tuesday', '09:00', '12:00'),
(11, N'Monday', '14:00', '17:00'), (11, N'Wednesday', '10:00', '13:00'),
(12, N'Thursday', '09:00', '12:00'), (12, N'Friday', '15:00', '18:00'),

-- Orthopedics 
(13, N'Sunday', '08:00', '11:00'), (13, N'Wednesday', '13:00', '16:00'),
(14, N'Monday', '09:00', '12:00'), (14, N'Thursday', '14:00', '17:00'),
(15, N'Tuesday', '10:00', '13:00'), (15, N'Saturday', '11:00', '14:00'),

-- ENT 
(16, N'Sunday', '10:00', '13:00'), (16, N'Thursday', '15:00', '18:00'),
(17, N'Monday', '08:00', '11:00'), (17, N'Wednesday', '14:00', '17:00'),
(18, N'Tuesday', '11:00', '14:00'), (18, N'Friday', '09:00', '12:00'),

-- Ophthalmology 
(19, N'Sunday', '09:00', '12:00'), (19, N'Tuesday', '13:00', '16:00'),
(20, N'Monday', '10:00', '13:00'), (20, N'Thursday', '14:00', '17:00'),
(21, N'Wednesday', '11:00', '14:00'), (21, N'Saturday', '09:00', '12:00'),

-- Psychiatry 
(22, N'Sunday', '14:00', '17:00'), (22, N'Wednesday', '09:00', '12:00'),
(23, N'Monday', '13:00', '16:00'), (23, N'Thursday', '10:00', '13:00'),
(24, N'Tuesday', '15:00', '18:00'), (24, N'Friday', '11:00', '14:00'),

-- Urology 
(25, N'Sunday', '08:00', '11:00'), (25, N'Thursday', '12:00', '15:00'),
(26, N'Monday', '09:00', '12:00'), (26, N'Wednesday', '14:00', '17:00'),
(27, N'Tuesday', '10:00', '13:00'), (27, N'Saturday', '13:00', '16:00'),

-- Oncology 
(28, N'Sunday', '10:00', '13:00'), (28, N'Wednesday', '15:00', '18:00'),
(29, N'Monday', '11:00', '14:00'), (29, N'Thursday', '09:00', '12:00'),
(30, N'Tuesday', '14:00', '17:00'), (30, N'Friday', '10:00', '13:00'),

-- General Surgery 
(31, N'Sunday', '09:00', '12:00'), (31, N'Tuesday', '13:00', '16:00'),
(32, N'Monday', '08:00', '11:00'), (32, N'Wednesday', '14:00', '17:00'),
(33, N'Thursday', '10:00', '13:00'), (33, N'Saturday', '11:00', '14:00'),

-- Radiology
(34, N'Sunday', '11:00', '14:00'), (34, N'Thursday', '08:00', '11:00'),
(35, N'Monday', '12:00', '15:00'), (35, N'Wednesday', '09:00', '12:00'),
(36, N'Tuesday', '13:00', '16:00'), (36, N'Friday', '10:00', '13:00');

INSERT INTO patients (
    username, password, first_name, last_name, gender, date_of_birth, 
    phone, email, national_id, address, blood_type, 
    emergency_contact, emergency_phone
)
VALUES
('nancy01', 'Pass1234', 'Nancy', 'Ahmed', 'Female', '2002-11-04', 
 '01012345678', 'nancy@example.com', '30201122334455', 'Alexandria', 'A+', 
 'Mona Ahmed', '01234567890'),

('omar_92', 'Omar2025x', 'Omar', 'Hassan', 'Male', '1998-05-22', 
 '01098765432', 'omar.hassan@example.com', '29805223344556', 'Cairo', 'B+', 
 'Sara Hassan', '01123456789'),

('salma_k', 'SalmA123!', 'Salma', 'Khaled', 'Female', '2000-09-15', 
 '01156789012', 'salma.k@example.com', '30009152364789', 'Giza', 'O-', 
 'Khaled Youssef', '01099887766'),

('ahmed_ali', 'AliPass99', 'Ahmed', 'Ali', 'Male', '1995-03-10', 
 '01233445566', 'ahmed.ali@example.com', '29503102233445', 'Tanta', 'AB+', 
 'Mohamed Ali', '01077665544'),

('layla77', 'Layla2024A', 'Layla', 'Mostafa', 'Female', '2001-12-30', 
 '01566778899', 'layla.m@example.com', '30112303344556', 'Mansoura', 'A-', 
 'Hassan Mostafa', '01122334455');

INSERT INTO medical_history (patient_id, chronic_diseases, allergies, current_medications, past_surgeries, family_history, notes) VALUES
(1, N'Diabetes', N'Penicillin', N'Insulin', N'Appendectomy (2010)', N'Father has hypertension', N'Patient follows up monthly'),
(2, N'Asthma', N'Dust', N'Inhaler', NULL, N'Mother has asthma', N'Seasonal symptoms, controlled'),
(3, NULL, N'Peanuts', NULL, NULL, NULL, N'Healthy, no major history'),
(4, N'Hypertension', NULL, N'Amlodipine', NULL, N'Grandmother had stroke', N'Patient advised to reduce salt intake');

INSERT INTO appointments (patient_id, doctor_id, appointment_day, appointment_time, status) VALUES
(1, 1, N'Sunday', '10:00:00', N'Booked'),
(2, 2, N'Monday', '15:30:00', N'Booked'),
(3, 3, N'Tuesday', '09:15:00', N'Completed'),
(4, 4, N'Wednesday', '11:45:00', N'Cancelled'),
(1, 5, N'Thursday', '14:00:00', N'Pending');

ALTER TABLE lab_report_analysis
ALTER COLUMN result NVARCHAR(MAX);

ALTER TABLE medical_history
ALTER COLUMN notes NVARCHAR(MAX);

ALTER TABLE lab_report_analysis
ALTER COLUMN analysis_type NVARCHAR(50) NULL;

ALTER TABLE lab_report_analysis
ALTER COLUMN analysis_type NVARCHAR(100) NULL;

ALTER TABLE lab_report_analysis
ALTER COLUMN language NVARCHAR(50) NULL;
