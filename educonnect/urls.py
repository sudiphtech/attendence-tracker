from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path("", RedirectView.as_view(url="/analytics/dashboard/", permanent=False)),
    path('admin/', admin.site.urls),
    path('analytics/', include('analytics_app.urls')),
]
