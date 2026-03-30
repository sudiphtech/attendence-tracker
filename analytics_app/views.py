import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .models import AttendanceRecord, Student


PRESENT_STATUSES = {"present", "late", "left early", "excused"}
ABSENT_STATUSES = {"absent", "absnt"}
STUDENT_CSV_COLUMNS = [
    "Student_ID",
    "Full_Name",
    "Date_of_Birth",
    "Grade_Level",
    "Emergency_Contact",
    "Secondary_Contact",
]
CHATBOT_SYSTEM_PROMPT = (
    "You are EduConnect Assistant, a helpful general-purpose chatbot inside a student "
    "attendance application. Answer the user's question clearly and directly. If the "
    "question is about attendance or students, be especially practical. If you are not "
    "sure, say so instead of inventing facts."
)
WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"


def _project_csv_path(filename):
    return os.path.join(settings.BASE_DIR, "analytics_app", filename)


def _normalize_status(value):
    return str(value).strip().lower()


def _load_students_csv():
    df = _load_students_csv_raw()
    df = df.rename(
        columns={
            "Student_ID": "roll_number",
            "Full_Name": "name",
        }
    )
    df["roll_number"] = df["roll_number"].astype(str).str.strip()
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    return df[["roll_number", "name"]]


def _load_students_csv_raw():
    path = _project_csv_path("students.csv")
    df = pd.read_csv(path)
    if "Unnamed: 5" in df.columns and "Secondary_Contact" not in df.columns:
        df = df.rename(columns={"Unnamed: 5": "Secondary_Contact"})

    for column in STUDENT_CSV_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    return df[STUDENT_CSV_COLUMNS].fillna("")


def _save_students_csv_raw(df):
    path = _project_csv_path("students.csv")
    normalized = df.copy()
    for column in STUDENT_CSV_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
    normalized = normalized[STUDENT_CSV_COLUMNS].fillna("")
    normalized.to_csv(path, index=False)


def _load_attendance_csv():
    path = _project_csv_path("final_attendance_dataset.csv")
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Student_ID": "roll_number",
            "Date": "date",
            "Attendance_Status": "attendance_status",
        }
    )
    df["roll_number"] = df["roll_number"].astype(str).str.strip()
    df["attendance_status"] = df["attendance_status"].map(_normalize_status)
    df["present"] = df["attendance_status"].isin(PRESENT_STATUSES)
    df["absent"] = df["attendance_status"].isin(ABSENT_STATUSES)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[["roll_number", "date", "attendance_status", "present", "absent"]]


def _dashboard_stats_from_csv():
    students_df = _load_students_csv()
    attendance_df = _load_attendance_csv()

    attendance_summary = (
        attendance_df.groupby("roll_number")
        .agg(
            total_days=("date", "count"),
            present_days=("present", "sum"),
        )
        .reset_index()
    )

    merged = students_df.merge(attendance_summary, on="roll_number", how="left").fillna(
        {"total_days": 0, "present_days": 0}
    )
    merged["total_days"] = merged["total_days"].astype(int)
    merged["present_days"] = merged["present_days"].astype(int)
    merged["attendance_percentage"] = np.where(
        merged["total_days"] > 0,
        (merged["present_days"] / merged["total_days"] * 100).round(2),
        0.0,
    )

    return merged.to_dict("records")


def _build_feature_summary(attendance_df):
    summary = (
        attendance_df.groupby("roll_number")
        .agg(
            total_days=("date", "count"),
            present_days=("present", "sum"),
            late_count=("attendance_status", lambda values: (values == "late").sum()),
        )
        .reset_index()
    )
    return {
        row["roll_number"]: {
            "total_days": int(row["total_days"]),
            "present_days": int(row["present_days"]),
            "late_count": int(row["late_count"]),
        }
        for _, row in summary.iterrows()
    }


def _student_defaults(row, feature_summary):
    roll_number = str(row["roll_number"]).strip()
    stats = feature_summary.get(
        roll_number,
        {"total_days": 0, "present_days": 0, "late_count": 0},
    )
    total_days = stats["total_days"]
    present_days = stats["present_days"]
    attendance_pct = round((present_days / total_days) * 100, 2) if total_days else 0.0

    return {
        "name": str(row["name"]).strip(),
        "attendance_pct": attendance_pct,
        "classes_recent": total_days,
        "late_count": stats["late_count"],
        "assignment_rate": 0.0,
        "engagement": 0.0,
    }


def _find_student_features_by_name(student_name):
    students_df = _load_students_csv()
    attendance_df = _load_attendance_csv()
    feature_summary = _build_feature_summary(attendance_df)

    lookup_name = student_name.strip().lower()
    if not lookup_name:
        raise ValueError("Please enter a student name.")

    exact_matches = students_df[students_df["name"].str.lower() == lookup_name]
    if exact_matches.empty:
        partial_matches = students_df[students_df["name"].str.lower().str.contains(lookup_name)]
    else:
        partial_matches = exact_matches

    if partial_matches.empty:
        raise ValueError(f'No student found for name "{student_name}".')

    if len(partial_matches) > 1:
        names = ", ".join(
            f'{row["name"]} ({row["roll_number"]})'
            for _, row in partial_matches.head(5).iterrows()
        )
        raise ValueError(
            "Multiple students matched that name. Please be more specific: "
            f"{names}"
        )

    student = partial_matches.iloc[0]
    stats = feature_summary.get(
        student["roll_number"],
        {"total_days": 0, "present_days": 0, "late_count": 0},
    )
    total_days = stats["total_days"]
    present_days = stats["present_days"]
    attendance_pct = round((present_days / total_days) * 100, 2) if total_days else 0.0

    return {
        "name": student["name"],
        "roll_number": student["roll_number"],
        "attendance_pct": attendance_pct,
        "classes_recent": total_days,
        "late_count": stats["late_count"],
    }


def _student_payload_from_row(row):
    return {
        "roll_number": str(row.get("Student_ID", "")).strip(),
        "name": str(row.get("Full_Name", "")).strip(),
        "date_of_birth": "" if pd.isna(row.get("Date_of_Birth")) else str(row.get("Date_of_Birth")).strip(),
        "grade_level": "" if pd.isna(row.get("Grade_Level")) else str(row.get("Grade_Level")).strip(),
        "emergency_contact": "" if pd.isna(row.get("Emergency_Contact")) else str(row.get("Emergency_Contact")).strip(),
        "secondary_contact": "" if pd.isna(row.get("Secondary_Contact")) else str(row.get("Secondary_Contact")).strip(),
    }


def _extract_response_text(response_json):
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        parts = []
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = str(content.get("text", "")).strip()
                if text:
                    parts.append(text)
        if parts:
            return "\n\n".join(parts)
    return ""


def _strip_html(value):
    return re.sub(r"<[^>]+>", "", str(value or "")).strip()


def _fetch_wikipedia_context(query, limit=3):
    params = urllib.parse.urlencode({"q": query, "limit": limit})
    url = f"{WIKIPEDIA_SEARCH_URL}?{params}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "EduConnect/1.0 (Wikipedia knowledge lookup)"
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Wikipedia could not be reached right now. Check your internet connection."
        ) from exc

    pages = []
    for page in data.get("pages", [])[:limit]:
        title = str(page.get("title", "")).strip()
        key = str(page.get("key", "")).strip()
        excerpt = _strip_html(page.get("excerpt", ""))
        description = str(page.get("description", "") or "").strip()
        if not title:
            continue
        pages.append(
            {
                "title": title,
                "description": description,
                "excerpt": excerpt,
                "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(key or title.replace(' ', '_'))}",
            }
        )
    return pages


def _format_wikipedia_context(pages):
    lines = []
    for index, page in enumerate(pages, start=1):
        pieces = [f"{index}. {page['title']}"]
        if page["description"]:
            pieces.append(f"Description: {page['description']}")
        if page["excerpt"]:
            pieces.append(f"Excerpt: {page['excerpt']}")
        pieces.append(f"URL: {page['url']}")
        lines.append("\n".join(pieces))
    return "\n\n".join(lines)


def _build_wikipedia_fallback_answer(query, pages):
    if not pages:
        return {
            "answer": (
                f'I could not find a matching Wikipedia result for "{query}". '
                "Try a more specific question or set OPENAI_API_KEY for a broader chatbot answer."
            ),
            "model": "wikipedia-search",
            "sources": [],
        }

    top = pages[0]
    answer_parts = [f"According to Wikipedia, {top['title']}"]
    if top["description"]:
        answer_parts.append(f"is {top['description'].lower()}")
    if top["excerpt"]:
        answer_parts.append(top["excerpt"])

    answer = ". ".join(part.rstrip(".") for part in answer_parts if part).strip() + "."
    if len(pages) > 1:
        answer += " I also found related pages you can open for more detail."

    return {
        "answer": answer,
        "model": "wikipedia-search",
        "sources": [{"title": page["title"], "url": page["url"]} for page in pages],
    }


def _chatbot_reply(messages):
    last_user_message = next(
        (message["content"] for message in reversed(messages) if message.get("role") == "user"),
        "",
    )
    wikipedia_pages = _fetch_wikipedia_context(last_user_message, limit=3) if last_user_message else []
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _build_wikipedia_fallback_answer(last_user_message, wikipedia_pages)

    model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    input_messages = list(messages)
    if wikipedia_pages:
        input_messages.append(
            {
                "role": "developer",
                "content": (
                    "Use the following Wikipedia search context when it is relevant. "
                    "Do not claim certainty beyond this context. Cite the article titles naturally in the answer.\n\n"
                    f"{_format_wikipedia_context(wikipedia_pages)}"
                ),
            }
        )
    payload = {
        "model": model,
        "instructions": CHATBOT_SYSTEM_PROMPT,
        "input": input_messages,
        "text": {"format": {"type": "text"}},
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        try:
            error_json = json.loads(details)
            message = error_json.get("error", {}).get("message") or details
        except json.JSONDecodeError:
            message = details or str(exc)
        raise RuntimeError(f"Chatbot request failed: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "The chatbot could not reach the AI service. Check your internet connection."
        ) from exc

    answer = _extract_response_text(response_json)
    if not answer:
        raise RuntimeError("The chatbot did not return any text.")
    return {
        "answer": answer,
        "model": response_json.get("model", model),
        "sources": [{"title": page["title"], "url": page["url"]} for page in wikipedia_pages],
    }


@csrf_exempt
def predict_risk(request):
    model_path = os.path.join(settings.BASE_DIR, "analytics_app", "model.pkl")
    model = joblib.load(model_path)
    if request.method == "POST":
        data = json.loads(request.body)
        features = np.array(
            [[data["attendance_pct"], data["classes_recent"], data["late_count"]]]
        )
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        result = "At Risk" if prediction == 1 else "Safe"
        return JsonResponse(
            {
                "prediction": result,
                "risk_probability": float(probability),
            }
        )


@csrf_exempt
def students_api(request):
    if request.method == "GET":
        try:
            students_df = _load_students_csv_raw()
        except FileNotFoundError as exc:
            return JsonResponse({"error": "CSV file not found", "path": str(exc)}, status=404)

        students = [_student_payload_from_row(row) for _, row in students_df.iterrows()]
        return JsonResponse({"count": len(students), "students": students})

    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Request body must be valid JSON"}, status=400)

    roll_number = str(data.get("roll_number", "")).strip().upper()
    name = str(data.get("name", "")).strip()
    date_of_birth = str(data.get("date_of_birth", "")).strip()
    grade_level = str(data.get("grade_level", "")).strip()
    emergency_contact = str(data.get("emergency_contact", "")).strip()
    secondary_contact = str(data.get("secondary_contact", "")).strip()

    if not roll_number:
        return JsonResponse({"error": "roll_number is required"}, status=400)
    if not name:
        return JsonResponse({"error": "name is required"}, status=400)

    try:
        students_df = _load_students_csv_raw()
    except FileNotFoundError as exc:
        return JsonResponse({"error": "CSV file not found", "path": str(exc)}, status=404)

    existing_roll_numbers = (
        students_df["Student_ID"].fillna("").astype(str).str.strip().str.upper()
        if "Student_ID" in students_df.columns
        else pd.Series(dtype=str)
    )
    if existing_roll_numbers.eq(roll_number).any():
        return JsonResponse(
            {"error": f"Student with roll_number {roll_number} already exists"},
            status=409,
        )

    new_row = {
        "Student_ID": roll_number,
        "Full_Name": name,
        "Date_of_Birth": date_of_birth,
        "Grade_Level": grade_level,
        "Emergency_Contact": emergency_contact,
        "Secondary_Contact": secondary_contact,
    }
    students_df = pd.concat([students_df, pd.DataFrame([new_row])], ignore_index=True)
    _save_students_csv_raw(students_df)

    created_student = _student_payload_from_row(new_row)
    return JsonResponse({"message": "Student created", "student": created_student}, status=201)


def chatbot_page(request):
    return render(request, "analytics_app/chatbot.html")


@csrf_exempt
def chatbot_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Request body must be valid JSON"}, status=400)

    raw_messages = data.get("messages", [])
    if not isinstance(raw_messages, list) or not raw_messages:
        return JsonResponse({"error": "messages must be a non-empty list"}, status=400)

    messages = []
    for message in raw_messages[-12:]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content})

    if not messages or messages[-1]["role"] != "user":
        return JsonResponse({"error": "The final message must be a user message"}, status=400)

    try:
        result = _chatbot_reply(messages)
    except RuntimeError as exc:
        return JsonResponse({"error": str(exc)}, status=503)

    return JsonResponse(result)


def dashboard(request):
    query = request.GET.get("q", "").strip()
    page_number = request.GET.get("page", 1)

    try:
        stats = _dashboard_stats_from_csv()
    except Exception:
        students = Student.objects.all()
        stats = []
        for student in students:
            records = student.attendance.order_by("date")
            total = records.count()
            present = records.filter(present=True).count()
            percentage = round((present / total * 100), 2) if total > 0 else 0.0
            stats.append(
                {
                    "name": student.name,
                    "roll_number": student.roll_number,
                    "attendance_percentage": percentage,
                    "total_days": total,
                    "present_days": present,
                }
            )

    if query:
        query_lower = query.lower()
        stats = [
            student
            for student in stats
            if query_lower in student["name"].lower()
            or query_lower in student["roll_number"].lower()
        ]

    paginator = Paginator(stats, 100)
    page_obj = paginator.get_page(page_number)

    return render(
        request,
        "analytics_app/dashboard.html",
        {
            "students": page_obj.object_list,
            "page_obj": page_obj,
            "query": query,
            "total_students": len(stats),
        },
    )


def import_students(request):
    try:
        students_df = _load_students_csv()
        attendance_df = _load_attendance_csv()
    except FileNotFoundError as exc:
        return JsonResponse({"error": "CSV file not found", "path": str(exc)}, status=404)

    feature_summary = _build_feature_summary(attendance_df)
    imported = 0
    for _, row in students_df.iterrows():
        _, created = Student.objects.update_or_create(
            roll_number=row["roll_number"],
            defaults=_student_defaults(row, feature_summary),
        )
        if created:
            imported += 1

    return JsonResponse({"students_imported": imported, "students_total": len(students_df)})


def predict_form(request):
    model_path = os.path.join(settings.BASE_DIR, "analytics_app", "model.pkl")
    model = joblib.load(model_path)
    context = {
        "student_name": "",
        "prediction": None,
        "risk_probability": None,
        "error": None,
        "selected_student": None,
        "used_features": None,
    }
    if request.method == "POST":
        try:
            student_name = request.POST.get("student_name", "").strip()
            context["student_name"] = student_name
            student_features = _find_student_features_by_name(student_name)
            features = np.array(
                [[
                    student_features["attendance_pct"],
                    student_features["classes_recent"],
                    student_features["late_count"],
                ]]
            )
            prediction = model.predict(features)[0]
            probability = float(model.predict_proba(features)[0][1])
            context["prediction"] = "At Risk" if prediction == 1 else "Safe"
            context["risk_probability"] = probability
            context["selected_student"] = {
                "name": student_features["name"],
                "roll_number": student_features["roll_number"],
            }
            context["used_features"] = {
                "attendance_pct": student_features["attendance_pct"],
                "classes_recent": student_features["classes_recent"],
                "late_count": student_features["late_count"],
            }
        except Exception as exc:
            context["error"] = str(exc)
    return render(request, "analytics_app/predict_form.html", context)


def import_csv(request):
    return import_students(request)


def import_attendance_csv(request):
    try:
        students_df = _load_students_csv()
        attendance_df = _load_attendance_csv()
    except FileNotFoundError as exc:
        return JsonResponse({"error": "CSV file not found", "path": str(exc)}, status=404)

    student_lookup = {}
    for _, row in students_df.iterrows():
        student = Student.objects.filter(roll_number=row["roll_number"]).first()
        if student:
            student_lookup[row["roll_number"]] = student

    imported = 0
    for _, row in attendance_df.iterrows():
        student = student_lookup.get(row["roll_number"])
        if not student or pd.isna(row["date"]):
            continue

        _, created = AttendanceRecord.objects.update_or_create(
            student=student,
            date=row["date"].date(),
            defaults={"present": bool(row["present"])},
        )
        if created:
            imported += 1

    return JsonResponse(
        {
            "attendance_imported": imported,
            "attendance_total": len(attendance_df),
        }
    )


def import_all(request):
    results = {
        "students_csv": {"imported": 0, "total": 0, "error": None},
        "attendance_csv": {"imported": 0, "total": 0, "error": None},
        "model_trained": False,
        "error": None,
    }

    try:
        students_df = _load_students_csv()
        attendance_df = _load_attendance_csv()
        results["students_csv"]["total"] = len(students_df)
        results["attendance_csv"]["total"] = len(attendance_df)
    except Exception as exc:
        results["error"] = str(exc)
        return JsonResponse(results, status=400)

    feature_summary = _build_feature_summary(attendance_df)
    imported_students = 0
    for _, row in students_df.iterrows():
        _, created = Student.objects.update_or_create(
            roll_number=row["roll_number"],
            defaults=_student_defaults(row, feature_summary),
        )
        if created:
            imported_students += 1
    results["students_csv"]["imported"] = imported_students

    student_lookup = {
        student.roll_number: student for student in Student.objects.filter(
            roll_number__in=students_df["roll_number"].tolist()
        )
    }
    imported_attendance = 0
    for _, row in attendance_df.iterrows():
        student = student_lookup.get(row["roll_number"])
        if not student or pd.isna(row["date"]):
            continue

        _, created = AttendanceRecord.objects.update_or_create(
            student=student,
            date=row["date"].date(),
            defaults={"present": bool(row["present"])},
        )
        if created:
            imported_attendance += 1
    results["attendance_csv"]["imported"] = imported_attendance

    try:
        from .train_model import train_model

        train_model()
        results["model_trained"] = True
    except Exception as exc:
        results["error"] = f"Model training failed: {exc}"

    return JsonResponse(results)
