const flashMessage = (message, catetgory = "info") => {
  const flashContainer = document.getElementById("flash-container");
  let newToast = document.createElement("div");
  newToast.setAttribute(
    "class",
    `toast align-items-center text-white bg-${catetgory} border-0`
  );
  newToast.innerHTML = `
            <div class="d-flex">
              <div class="toast-body">${message}</div>
              <button
                type="button"
                class="btn-close btn-close-white me-2 m-auto"
                data-bs-dismiss="toast"
                aria-label="Close"
              ></button>
            </div>
            `;
  flashContainer.appendChild(newToast);
  newToast = new bootstrap.Toast(newToast);
  newToast.show();
};
