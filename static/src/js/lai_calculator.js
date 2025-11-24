/** @odoo-module */

$(document).on('submit', '.lai-upload-form', function () {
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
    const $fields = $('#custom-calibration-fields');
    if ($checkbox.length && $checkbox.is(':checked')) {
        $fields.show();
    }
});