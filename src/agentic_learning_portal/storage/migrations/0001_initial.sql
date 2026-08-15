-- 0001_initial: create the portal storage schema and seed the three roles.
--
-- A user can hold several roles at once (admin, teacher, student): roles and
-- users are linked through the user_roles join table rather than a single
-- column on users.

-- Roles: the fixed set of roles in the system.
CREATE TABLE roles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

-- Users: a person who can log in.
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT UNIQUE,
    created_at TEXT NOT NULL
);

-- A user can hold many roles and a role can belong to many users.
CREATE TABLE user_roles (
    user_id INTEGER NOT NULL REFERENCES users(id),
    role_id INTEGER NOT NULL REFERENCES roles(id),
    PRIMARY KEY (user_id, role_id)
);

-- Tasks: generated tasks persisted from the generator output.
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    text TEXT NOT NULL,
    complexity TEXT NOT NULL,
    correct_answer TEXT NOT NULL,
    solution TEXT NOT NULL
);

-- Assignments: a collection of tasks assigned by a teacher/admin to a student.
CREATE TABLE assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_by INTEGER NOT NULL REFERENCES users(id),
    assigned_to INTEGER REFERENCES users(id),
    created_at TEXT NOT NULL
);

-- Ordered many-to-many link between assignments and tasks.
CREATE TABLE assignment_tasks (
    assignment_id INTEGER NOT NULL REFERENCES assignments(id),
    task_id INTEGER NOT NULL REFERENCES tasks(id),
    position INTEGER NOT NULL,
    PRIMARY KEY (assignment_id, task_id)
);

-- A student's attempt at an assignment.
CREATE TABLE attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment_id INTEGER NOT NULL REFERENCES assignments(id),
    student_id INTEGER NOT NULL REFERENCES users(id),
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

-- A scored answer for one task within an attempt.
CREATE TABLE attempt_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id INTEGER NOT NULL REFERENCES attempts(id),
    task_id INTEGER NOT NULL REFERENCES tasks(id),
    given_answer TEXT,
    expected_answer TEXT,
    is_correct INTEGER,
    score REAL,
    detail TEXT NOT NULL DEFAULT ''
);

-- Seed the three built-in roles.
INSERT INTO roles (name) VALUES ('admin'), ('teacher'), ('student');