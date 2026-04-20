/** @odoo-module */

function validateCalibration() {
    const useCustom = document.getElementById("use_custom")?.checked;
    if (!useCustom) {
        return { valid: true };
    }

    const getVal = (name) => {
        const el = document.querySelector(`[name="${name}"]`);
        return el ? parseFloat(el.value) : NaN;
    };

    const hueCenter = getVal("custom_green_hue_center");
    const hueWidth = getVal("custom_green_hue_width");
    const laiMin = getVal("custom_lai_min");
    const laiMax = getVal("custom_lai_max");

    if (Number.isNaN(hueCenter) || hueCenter < 0 || hueCenter > 1) {
        return { valid: false, message: "Green Hue Center must be between 0 and 1." };
    }
    if (Number.isNaN(hueWidth) || hueWidth < 0.1 || hueWidth > 10) {
        return { valid: false, message: "Hue Sensitivity must be between 0.1 and 10." };
    }
    if (Number.isNaN(laiMin) || laiMin < 0 || laiMin > 5.9) {
        return { valid: false, message: "Min LAI must be between 0 and 5.9." };
    }
    if (Number.isNaN(laiMax) || laiMax < 0.1 || laiMax > 10) {
        return { valid: false, message: "Max LAI must be between 0.1 and 10." };
    }
    if (laiMin >= laiMax) {
        return { valid: false, message: "Min LAI must be less than Max LAI." };
    }

    return { valid: true };
}

function updateSamInfo() {
    const samSelected = document.getElementById("method_sam")?.checked;
    const info = document.getElementById("sam-requirements-info");
    if (!info) {
        return;
    }

    if (samSelected) {
        $(info).stop(true, true).slideDown(180);
    } else {
        $(info).stop(true, true).slideUp(180);
    }
}

function formatLocalDatetime() {
    const elements = document.querySelectorAll(".lai-local-datetime");

    elements.forEach((span) => {
        const utc = span.dataset.utc;
        if (!utc) {
            return;
        }

        const date = new Date(utc);
        const formatted = date.toLocaleString([], {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        });
        span.textContent = formatted;
    });
}

$(document).on("change", ".lai-file-input", function () {
    const file = this.files?.[0];
    if (!file || !file.type.startsWith("image/")) {
        return;
    }

    const $form = $(this).closest("form");
    const $previewContainer = $form.find("#image-preview-container");
    const $previewImg = $form.find("#image-preview");

    if (!$previewContainer.length || !$previewImg.length) {
        return;
    }

    const reader = new FileReader();
    reader.onload = function (event) {
        $previewImg.attr("src", event.target.result);
        $previewContainer.stop(true, true).fadeIn(180);
    };
    reader.readAsDataURL(file);
});

$(document).on("submit", ".lai-upload-form", function (event) {
    const validation = validateCalibration();
    if (!validation.valid) {
        event.preventDefault();
        alert(`Validation Error: ${validation.message}`);
        return false;
    }

    const $btn = $(this).find(".lai-submit-btn");
    $btn.html("<span>Выполняется расчёт…</span>").prop("disabled", true);
    return true;
});

$(document).on("change", "#use_custom", function () {
    const $fields = $("#custom-calibration-fields");
    if (this.checked) {
        $fields.stop(true, true).slideDown(180);
    } else {
        $fields.stop(true, true).slideUp(180);
    }
});

$(document).on("change", 'input[name="segmentation_method"]', function () {
    updateSamInfo();
});

$(document).ready(function () {
    const $checkbox = $("#use_custom");
    if ($checkbox.length && $checkbox.is(":checked")) {
        $("#custom-calibration-fields").show();
    }
    formatLocalDatetime();
    updateSamInfo();
});
