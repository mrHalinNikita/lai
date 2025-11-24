/** @odoo-module */

function validateCalibration() {
    const useCustom = document.getElementById('use_custom')?.checked;
    if (!useCustom) return { valid: true };

    const getVal = (name) => {
        const el = document.querySelector(`[name="${name}"]`);
        return el ? parseFloat(el.value) : NaN;
    };

    const hueCenter = getVal('custom_green_hue_center');
    const hueWidth = getVal('custom_green_hue_width');
    const laiMin = getVal('custom_lai_min');
    const laiMax = getVal('custom_lai_max');

    if (isNaN(hueCenter) || hueCenter < 0 || hueCenter > 1) {
        return { valid: false, message: "Green Hue Center must be between 0 and 1." };
    }
    if (isNaN(hueWidth) || hueWidth < 0.1 || hueWidth > 10) {
        return { valid: false, message: "Hue Sensitivity must be between 0.1 and 10." };
    }
    if (isNaN(laiMin) || laiMin < 0 || laiMin > 5.9) {
        return { valid: false, message: "Min LAI must be between 0 and 5.9." };
    }
    if (isNaN(laiMax) || laiMax < 0.1 || laiMax > 10) {
        return { valid: false, message: "Max LAI must be between 0.1 and 10." };
    }
    if (laiMin >= laiMax) {
        return { valid: false, message: "Min LAI must be less than Max LAI." };
    }

    return { valid: true };
}

$(document).on('change', '.lai-file-input', function () {
    const file = this.files?.[0];
    if (!file || !file.type.startsWith('image/')) return;

    const $form = $(this).closest('form');
    const $previewContainer = $form.find('#image-preview-container');
    const $previewImg = $form.find('#image-preview');

    if (!$previewContainer.length || !$previewImg.length) {
        return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
        $previewImg.attr('src', e.target.result);
        $previewContainer.show();
    };
    reader.readAsDataURL(file);
});

$(document).on('submit', '.lai-upload-form', function (e) {
    const validation = validateCalibration();
    if (!validation.valid) {
        e.preventDefault();
        alert('Validation Error: ' + validation.message);
        return false;
    }

    const $btn = $(this).find('.lai-submit-btn');
    $btn.html('<span>Calculating…</span>').prop('disabled', true);
});

$(document).on('change', '#use_custom', function () {
    const $fields = $('#custom-calibration-fields');
    if (this.checked) {
        $fields.slideDown(200);
    } else {
        $fields.slideUp(200);
    }
});

$(document).ready(function () {
    const $checkbox = $('#use_custom');
    if ($checkbox.length && $checkbox.is(':checked')) {
        $('#custom-calibration-fields').show();
    }
});