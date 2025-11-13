/**
 * WellbeingAI Main JavaScript
 * Handles common functionality across all pages
 */

// Global variables
let currentUser = {
    role: 'employee',
    id: 'web_user_' + Date.now()
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadUserPreferences();
});

/**
 * Initialize the application
 */
function initializeApp() {
    console.log('🚀 Initializing WellbeingAI Web Application');

    // Set up CSRF protection for forms
    setupCSRFProtection();

    // Initialize tooltips
    initializeTooltips();

    // Set up real-time updates if supported
    setupRealTimeUpdates();

    // Check for browser compatibility
    checkBrowserCompatibility();

    // Initialize theme
    initializeTheme();

    console.log('✅ Application initialized');
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Handle online/offline status
    window.addEventListener('online', handleOnlineStatus);
    window.addEventListener('offline', handleOfflineStatus);

    // Handle visibility change (tab switching)
    document.addEventListener('visibilitychange', handleVisibilityChange);

    // Handle before unload
    window.addEventListener('beforeunload', handleBeforeUnload);

    // Handle resize for responsive charts
    window.addEventListener('resize', handleWindowResize);

    // Set up keyboard shortcuts
    setupKeyboardShortcuts();
}

/**
 * Set up CSRF protection for AJAX requests
 */
function setupCSRFProtection() {
    // Get CSRF token from meta tag if available
    const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');

    if (csrfToken) {
        // Add CSRF token to all AJAX requests
        const originalFetch = window.fetch;
        window.fetch = function(...args) {
            if (args[1] && typeof args[1] === 'object') {
                args[1].headers = {
                    ...args[1].headers,
                    'X-CSRF-Token': csrfToken
                };
            }
            return originalFetch.apply(this, args);
        };
    }
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Set up real-time updates using Server-Sent Events or WebSockets
 */
function setupRealTimeUpdates() {
    // Check if the browser supports Server-Sent Events
    if (typeof(EventSource) !== "undefined") {
        try {
            const eventSource = new EventSource('/api/events');

            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleRealTimeUpdate(data);
            };

            eventSource.onerror = function() {
                console.log('Real-time updates not available, falling back to polling');
                setupPollingUpdates();
            };
        } catch (error) {
            console.log('Server-Sent Events not supported, using polling');
            setupPollingUpdates();
        }
    } else {
        setupPollingUpdates();
    }
}

/**
 * Set up polling for updates when real-time is not available
 */
function setupPollingUpdates() {
    // Poll for updates every 30 seconds
    setInterval(async () => {
        if (document.visibilityState === 'visible') {
            try {
                const response = await fetch('/api/health');
                if (response.ok) {
                    // Could check for new data here
                    console.log('Health check passed');
                }
            } catch (error) {
                console.log('Health check failed:', error);
            }
        }
    }, 30000);
}

/**
 * Handle real-time updates
 */
function handleRealTimeUpdate(data) {
    console.log('Real-time update received:', data);

    switch(data.type) {
        case 'risk_alert':
            showRiskAlert(data);
            break;
        case 'intervention_recommendation':
            showInterventionNotification(data);
            break;
        case 'checkin_reminder':
            showCheckinReminder(data);
            break;
        case 'system_status':
            updateSystemStatus(data);
            break;
        default:
            console.log('Unknown update type:', data.type);
    }
}

/**
 * Show risk alert notification
 */
function showRiskAlert(data) {
    const alertHtml = `
        <div class="alert alert-${data.severity || 'warning'} alert-dismissible fade show" role="alert">
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>Risk Alert:</strong> ${data.message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    // Insert at the top of the main content
    const mainContent = document.querySelector('main');
    if (mainContent) {
        mainContent.insertAdjacentHTML('afterbegin', alertHtml);
    }

    // Auto-dismiss after 10 seconds
    setTimeout(() => {
        const alert = document.querySelector('.alert');
        if (alert) {
            alert.remove();
        }
    }, 10000);
}

/**
 * Show intervention notification
 */
function showInterventionNotification(data) {
    // Use browser notification if permission granted
    if (Notification.permission === 'granted') {
        new Notification('WellbeingAI Intervention', {
            body: data.message,
            icon: '/static/images/wellbeing-icon.png'
        });
    }

    // Also show in-app notification
    showToast('New Intervention Available', data.message, 'info');
}

/**
 * Show check-in reminder
 */
function showCheckinReminder(data) {
    showToast('Daily Check-in Reminder', 'Don\'t forget to complete your daily wellbeing check-in!', 'primary');
}

/**
 * Update system status
 */
function updateSystemStatus(data) {
    const statusIndicator = document.getElementById('systemStatus');
    if (statusIndicator) {
        statusIndicator.className = `badge bg-${data.status === 'healthy' ? 'success' : 'warning'}`;
        statusIndicator.textContent = data.status;
    }
}

/**
 * Show toast notification
 */
function showToast(title, message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);
    }

    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    <strong>${title}</strong><br>
                    <small>${message}</small>
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

    toastContainer.insertAdjacentHTML('beforeend', toastHtml);

    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();

    // Remove from DOM after hiding
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

/**
 * Handle online status change
 */
function handleOnlineStatus() {
    showToast('Connection Restored', 'You are back online', 'success');
    // Refresh data when coming back online
    if (typeof refreshDashboard === 'function') {
        refreshDashboard();
    }
}

/**
 * Handle offline status change
 */
function handleOfflineStatus() {
    showToast('Connection Lost', 'You are currently offline. Some features may not be available.', 'warning');
}

/**
 * Handle visibility change (tab switching)
 */
function handleVisibilityChange() {
    if (document.visibilityState === 'visible') {
        // Tab became visible, refresh data if needed
        console.log('Tab became visible, checking for updates...');
        if (typeof refreshDashboard === 'function') {
            refreshDashboard();
        }
    }
}

/**
 * Handle before unload
 */
function handleBeforeUnload(event) {
    // Save any unsaved data
    saveUnsavedData();
}

/**
 * Handle window resize
 */
function handleWindowResize() {
    // Debounce chart resizing
    clearTimeout(window.resizeTimeout);
    window.resizeTimeout = setTimeout(() => {
        // Resize charts if they exist
        if (typeof resizeCharts === 'function') {
            resizeCharts();
        }
    }, 250);
}

/**
 * Set up keyboard shortcuts
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(event) {
        // Ctrl/Cmd + R to refresh dashboard
        if ((event.ctrlKey || event.metaKey) && event.key === 'r') {
            event.preventDefault();
            if (typeof refreshDashboard === 'function') {
                refreshDashboard();
            }
        }

        // Ctrl/Cmd + C to go to check-in
        if ((event.ctrlKey || event.metaKey) && event.key === 'c') {
            event.preventDefault();
            window.location.href = '/checkin';
        }

        // Escape to close modals
        if (event.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const bsModal = bootstrap.Modal.getInstance(modal);
                if (bsModal) {
                    bsModal.hide();
                }
            });
        }
    });
}

/**
 * Check browser compatibility
 */
function checkBrowserCompatibility() {
    const features = [
        { name: 'fetch', test: () => typeof fetch !== 'undefined' },
        { name: 'Promise', test: () => typeof Promise !== 'undefined' },
        { name: 'localStorage', test: () => typeof localStorage !== 'undefined' },
        { name: 'sessionStorage', test: () => typeof sessionStorage !== 'undefined' },
        { name: 'EventSource', test: () => typeof EventSource !== 'undefined' }
    ];

    const missingFeatures = features.filter(feature => !feature.test());

    if (missingFeatures.length > 0) {
        console.warn('Some features may not work properly in this browser:', missingFeatures.map(f => f.name));
        showToast('Browser Compatibility', 'Some features may not work properly in this browser. Please use a modern browser for the best experience.', 'warning');
    }
}

/**
 * Initialize theme (light/dark mode)
 */
function initializeTheme() {
    const savedTheme = localStorage.getItem('wellbeing-theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    // Theme toggle button
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
        updateThemeIcon(savedTheme);
    }
}

/**
 * Toggle between light and dark theme
 */
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';

    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('wellbeing-theme', newTheme);
    updateThemeIcon(newTheme);
}

/**
 * Update theme toggle icon
 */
function updateThemeIcon(theme) {
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        const icon = themeToggle.querySelector('i');
        if (icon) {
            icon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
        }
    }
}

/**
 * Load user preferences
 */
function loadUserPreferences() {
    try {
        const preferences = JSON.parse(localStorage.getItem('wellbeing-preferences') || '{}');

        // Apply saved preferences
        if (preferences.notifications !== undefined) {
            // Handle notification preferences
        }

        if (preferences.autoRefresh !== undefined) {
            // Handle auto-refresh preferences
        }

        console.log('User preferences loaded:', preferences);
    } catch (error) {
        console.log('Could not load user preferences:', error);
    }
}

/**
 * Save unsaved data before page unload
 */
function saveUnsavedData() {
    // Save any form data that might be in progress
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const formData = new FormData(form);
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }

        if (Object.keys(data).length > 0) {
            sessionStorage.setItem(`form-${form.id || 'unnamed'}`, JSON.stringify(data));
        }
    });
}

/**
 * Restore unsaved form data
 */
function restoreUnsavedData() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const savedData = sessionStorage.getItem(`form-${form.id || 'unnamed'}`);
        if (savedData) {
            try {
                const data = JSON.parse(savedData);
                Object.keys(data).forEach(key => {
                    const input = form.querySelector(`[name="${key}"]`);
                    if (input) {
                        input.value = data[key];
                    }
                });
            } catch (error) {
                console.log('Could not restore form data:', error);
            }
        }
    });
}

/**
 * Utility function to format dates
 */
function formatDate(date, options = {}) {
    const defaultOptions = {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    };

    return new Intl.DateTimeFormat('en-US', { ...defaultOptions, ...options }).format(new Date(date));
}

/**
 * Utility function to format relative time
 */
function formatRelativeTime(date) {
    const now = new Date();
    const diffInSeconds = Math.floor((now - new Date(date)) / 1000);

    if (diffInSeconds < 60) return 'just now';
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
    if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)} days ago`;

    return formatDate(date);
}

/**
 * Utility function to debounce function calls
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Utility function to throttle function calls
 */
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Export data as CSV
 */
function exportToCSV(data, filename) {
    const csvContent = "data:text/csv;charset=utf-8,"
        + data.map(row => row.join(",")).join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied', 'Text copied to clipboard', 'success');
    } catch (error) {
        console.error('Failed to copy text:', error);
        showToast('Error', 'Failed to copy text to clipboard', 'error');
    }
}

/**
 * Show loading spinner
 */
function showLoading(elementId, message = 'Loading...') {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="text-center py-4">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="text-muted mt-2">${message}</p>
            </div>
        `;
    }
}

/**
 * Hide loading spinner
 */
function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '';
    }
}

/**
 * Validate email format
 */
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

/**
 * Validate form inputs
 */
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return false;

    const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.classList.add('is-invalid');
            isValid = false;
        } else {
            input.classList.remove('is-invalid');
            input.classList.add('is-valid');
        }

        // Email validation
        if (input.type === 'email' && input.value && !isValidEmail(input.value)) {
            input.classList.add('is-invalid');
            isValid = false;
        }
    });

    return isValid;
}

/**
 * Reset form validation
 */
function resetFormValidation(formId) {
    const form = document.getElementById(formId);
    if (!form) return;

    const inputs = form.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        input.classList.remove('is-valid', 'is-invalid');
    });
}

// Make utility functions globally available
window.WellbeingAI = {
    showToast,
    showLoading,
    hideLoading,
    exportToCSV,
    copyToClipboard,
    formatDate,
    formatRelativeTime,
    validateForm,
    resetFormValidation,
    debounce,
    throttle
};

console.log('🎉 WellbeingAI utilities loaded and ready!');
