from datetime import datetime

from airflow.sdk import dag
from airflow.providers.smtp.operators.smtp import EmailOperator


@dag(
    dag_id="brainlens_email_test",
    schedule=None,
    start_date=datetime(2026, 8, 1),
    catchup=False,
    tags=["brainlens", "test"],
)
def brainlens_email_test():

    send_test_email = EmailOperator(
        task_id="send_test_email",
        to="prasadtest67@gmail.com",
        subject="[BrainLens] Airflow email test",
        html_content="""
        <h2>BrainLens Airflow Email Test</h2>

        <p>
            This email confirms that Airflow can successfully
            send email through the configured SMTP connection.
        </p>

        <p>
            SMTP connection:
            <strong>smtp_default</strong>
        </p>
        """,
        conn_id="smtp_default",
    )

    send_test_email


brainlens_email_test()