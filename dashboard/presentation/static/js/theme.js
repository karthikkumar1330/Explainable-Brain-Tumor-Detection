(function() {
    // 1. Guard against duplicate script executions/loads
    if (window.ThemeManager) {
        return;
    }

    // 2. Determine initial theme
    const storedTheme = localStorage.getItem('color-theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const currentTheme = storedTheme === 'dark' || (!storedTheme && prefersDark) ? 'dark' : 'light';
    
    // 3. Apply theme immediately to document element to prevent layout flashes
    applyThemeClass(currentTheme);

    function applyThemeClass(theme) {
        if (theme === 'dark') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }

    // 4. Main theme application function
    function applyTheme(theme) {
        applyThemeClass(theme);

        // Update charts if function exists
        if (typeof updateChartThemes === 'function') {
            updateChartThemes();
        }
    }

    function toggleTheme() {
        const isDark = document.documentElement.classList.contains('dark');
        const newTheme = isDark ? 'light' : 'dark';
        
        localStorage.setItem('color-theme', newTheme);
        document.cookie = "color-theme=" + newTheme + "; path=/; max-age=31536000; SameSite=Lax";
        
        applyTheme(newTheme);
    }

    // 5. Use event delegation to handle clicks on #theme-toggle.
    // This is bulletproof, avoids duplicate bindings, and works even if the button is dynamically re-inserted.
    document.addEventListener('click', function(e) {
        const toggleBtn = e.target.closest('#theme-toggle');
        if (toggleBtn) {
            e.preventDefault();
            toggleTheme();
        }
    });

    // 6. Sync across tabs using storage event
    window.addEventListener('storage', function(e) {
        if (e.key === 'color-theme') {
            applyTheme(e.newValue);
        }
    });

    // Expose utility globally
    window.ThemeManager = {
        applyTheme: applyTheme,
        toggleTheme: toggleTheme,
        currentTheme: function() {
            return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
        }
    };
})();
