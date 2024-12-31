class PhotoManager {
  constructor() {
    this.photos = [];
    this.initializeElements();
    this.initializeEventListeners();
    this.loadExistingPhotos();
  }

  initializeElements() {
    this.uploadArea = document.querySelector(".upload-area");
    this.uploadInterface = this.uploadArea.querySelector(".upload-interface");
    this.mainThumbnailOverlay = this.uploadArea.querySelector(
      ".main-thumbnail-overlay"
    );
    this.fileInput = document.getElementById("examPhotos");
    this.photosModal = new bootstrap.Modal(
      document.getElementById("photosModal")
    );
    this.photoDetailsModal = new bootstrap.Modal(
      document.getElementById("photoDetailsModal")
    );
  }

  handleUploadAreaClick() {
    document.getElementById("examPhotos").click();
  }

  initializeEventListeners() {
    // Drag and drop handlers
    this.uploadArea.addEventListener(
      "dragover",
      this.handleDragOver.bind(this)
    );
    this.uploadArea.addEventListener(
      "dragleave",
      this.handleDragLeave.bind(this)
    );
    this.uploadArea.addEventListener("drop", this.handleDrop.bind(this));

    // Click handlers
    this.uploadArea.addEventListener("click", this.handleUploadAreaClick);
    this.fileInput.addEventListener("change", this.handleFileSelect.bind(this));

    // View all photos button
    document.querySelector(".view-all-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      this.showPhotosModal();
    });

    // Add more photos button in modal
    document.getElementById("addMorePhotos").addEventListener("click", () => {
      this.photosModal.hide();
      this.fileInput.click();
    });
  }

  async loadExistingPhotos() {
    try {
      const response = await fetch(
        `/api/cases/${this.getCurrentCaseId()}/images`
      );
      if (!response.ok) throw new Error("Failed to fetch images");

      const data = await response.json();
      this.photos = data.images.map((img) => ({
        id: img.id,
        caseId: img.caseId,
        url: `/api/images/${img.id}`, // URL to fetch the image
        type: img.type || "",
        description: img.description || "",
        dateTaken: img.date_taken || new Date().toISOString().split("T")[0],
        isDefault: img.is_default || false,
        isAnalyzed: img.is_analyzed || false,
      }));

      this.updateDisplay();
    } catch (error) {
      console.error("Error loading existing photos:", error);
      this.showAlert("Failed to load existing photos", "danger");
    }
  }

  async handleFileSelect(event) {
    const files = event.target.files;
    await this.processFiles(files);
  }

  async handleDrop(event) {
    event.preventDefault();
    this.uploadArea.classList.remove("border-primary");
    const files = event.dataTransfer.files;
    await this.processFiles(files);
  }

  handleDragOver(event) {
    event.preventDefault();
    this.uploadArea.classList.add("border-primary");
  }

  handleDragLeave(event) {
    event.preventDefault();
    this.uploadArea.classList.remove("border-primary");
  }

  getCurrentCaseId() {
    return document.getElementById("case-title").dataset.caseId;
  }

  async processFiles(files) {
    this.mainThumbnailOverlay.classList.toggle("running");
    this.uploadArea.removeEventListener("click", this.handleUploadAreaClick);
    for (const file of files) {
      if (!file.type.startsWith("image/")) {
        this.showAlert("Only image files are allowed", "danger");
        continue;
      }
      try {
        const uploadedPhoto = await this.uploadPhoto(file);
        this.photos.push(uploadedPhoto);
        this.updateDisplay();
      } catch (error) {
        console.error("Upload error:", error);
        this.showAlert("Failed to upload image", "danger");
      }
    }
    this.mainThumbnailOverlay.classList.toggle("running");
    this.uploadArea.addEventListener("click", this.handleUploadAreaClick);
  }

  async uploadPhoto(file) {
    const formData = new FormData();
    formData.append("image", file);
    formData.append(
      "case_id",
      document.getElementById("case-title").dataset.caseId
    );

    const response = await fetch("/api/images/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Upload failed");
    }

    const data = await response.json();
    const imageUrl = await this.readFileAsDataURL(file);

    return {
      id: data.image_id,
      caseId: "",
      url: imageUrl,
      file: file,
      type: "",
      description: "",
      dateTaken: new Date().toISOString().split("T")[0],
      isDefault: this.photos.length === 0, // First photo becomes default
      isAnalyzed: true, // Assuming analysis happens on backend
    };
  }

  updateDisplay() {
    if (this.photos.length > 0) {
      this.showMainThumbnail();
      this.updatePhotosGrid();
    }
  }

  showMainThumbnail() {
    const defaultPhoto = this.photos.find((p) => p.isDefault) || this.photos[0];
    this.uploadInterface.classList.add("d-none");
    this.mainThumbnailOverlay.classList.remove("d-none");

    const thumbnail =
      this.mainThumbnailOverlay.querySelector(".main-thumbnail");
    thumbnail.src = defaultPhoto.url;

    const photoCount = this.mainThumbnailOverlay.querySelector(".photo-count");
    photoCount.textContent = `Photos & Files (${this.photos.length})`;
  }

  updatePhotosGrid() {
    const gridContainer = document.querySelector(".photos-grid");
    gridContainer.innerHTML = "";

    this.photos.forEach((photo) => {
      const photoElement = this.createPhotoElement(photo);
      gridContainer.appendChild(photoElement);
    });
  }

  createPhotoElement(photo) {
    const div = document.createElement("div");
    div.className = "photo-item";
    div.innerHTML = `
            <img src="${photo.url}" alt="Case photo">
            <button type="button" class="delete-btn" data-photo-id="${photo.id}">
                <i class="typcn typcn-times"></i>
            </button>
        `;

    // Add click handlers
    div.querySelector("img").addEventListener("click", () => {
      this.showPhotoDetails(photo);
    });

    div.querySelector(".delete-btn").addEventListener("click", async (e) => {
      e.stopPropagation();
      await this.deletePhoto(photo.id);
    });

    return div;
  }

  async deletePhoto(photoId) {
    try {
      const response = await fetch(`/api/images/${photoId}`, {
        method: "DELETE",
      });

      if (!response.ok) throw new Error("Delete failed");

      this.photos = this.photos.filter((p) => p.id !== photoId);

      // Update display
      if (this.photos.length === 0) {
        this.mainThumbnailOverlay.classList.add("d-none");
        this.uploadInterface.classList.remove("d-none");
      } else {
        this.updateDisplay();
      }

      this.updatePhotosGrid();
    } catch (error) {
      console.error("Delete error:", error);
      this.showAlert("Failed to delete image", "danger");
    }
  }

  showPhotoDetails(photo) {
    const modal = document.getElementById("photoDetailsModal");
    // Update modal content
    modal.querySelector("select").value = photo.type;
    modal.querySelector("select").dataset.imageId = photo.id;
    modal.querySelector("textarea").value = photo.description;
    modal.querySelector("textarea").dataset.imageId = photo.id;
    modal.querySelector('input[type="date"]').value = photo.dateTaken;
    modal.querySelector('input[type="date"]').dataset.imageId = photo.id;
    modal.querySelector("img").src = photo.url;

    // Update status indicators
    const indicators = modal.querySelectorAll(".indicator i");
    indicators[0].className = photo.isAnalyzed
      ? "typcn typcn-input-checked text-muted"
      : "typcn typcn-media-stop-outline text-muted";
    indicators[1].className = photo.isDefault
      ? "typcn typcn-input-checked text-success"
      : "typcn typcn-media-stop-outline text-muted";

    // Add event listeners for photo actions
    const actionButtons = modal.querySelectorAll(".photo-actions button");

    // Zoom button
    actionButtons[0].onclick = () => {
      window.open(photo.url, "_blank");
    };

    // Set as default button
    actionButtons[1].onclick = async () => {
      await this.setDefaultPhoto(photo.id, photo.caseId);
    };

    // Delete button
    actionButtons[2].onclick = async () => {
      await this.deletePhoto(photo.id);
      this.photoDetailsModal.hide();
    };

    this.photosModal.hide(); // Hide photos grid modal
    this.photoDetailsModal.show();
  }

  showPhotosModal() {
    this.updatePhotosGrid();
    this.photosModal.show();
  }

  async setDefaultPhoto(photoId, caseId) {
    this.photos.forEach((p) => (p.isDefault = p.id === photoId));
    this.updateDisplay();
    await updateImage(null, {
      id: photoId,
      field: "is_default",
      value: true,
      case_id: caseId,
    });
  }

  // Utility methods
  readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = (e) => reject(e);
      reader.readAsDataURL(file);
    });
  }

  showAlert(message, type = "info") {
    const alertDiv = document.createElement("div");
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
            ${message}
            <button type="button" class="close" data-dismiss="alert">
                <span>&times;</span>
            </button>
        `;

    this.uploadArea.parentNode.insertBefore(alertDiv, this.uploadArea);
    setTimeout(() => alertDiv.remove(), 5000);
  }
}

class NotesManager {
  constructor(caseId) {
    this.caseId = caseId;
    this.notesCount = 0;
    this.initializeElements();
    this.attachEventListeners();
    this.loadNotes();
  }

  initializeElements() {
    this.modal = $("#notesModal");
    this.notesContainer = this.modal.find(".notes-container");
    this.noteInput = $("#newNoteText");
    this.postButton = $("#postNoteBtn");
    this.notesCountBtn = $("#caseNotesBtn");
  }

  attachEventListeners() {
    this.postButton.on("click", () => this.postNote());
    this.noteInput.on("keydown", (e) => {
      if (e.ctrlKey && e.key === "Enter") {
        this.postNote();
      }
    });
  }

  async loadNotes() {
    try {
      const response = await fetch(`/api/cases/${this.caseId}/notes`);
      if (!response.ok) throw new Error("Failed to fetch notes");

      const data = await response.json();
      this.notesContainer.empty();

      data.notes.forEach((note) => this.addNoteToDisplay(note));
      this.updateNotesCount(data.notes.length);
    } catch (error) {
      console.error("Error loading notes:", error);
    }
  }

  async postNote() {
    const content = this.noteInput.val().trim();
    if (!content) return;

    try {
      const response = await fetch(`/api/cases/${this.caseId}/notes`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ content }),
      });

      if (!response.ok) throw new Error("Failed to post note");

      const note = await response.json();
      this.addNoteToDisplay(note);
      this.noteInput.val("");
      this.updateNotesCount(this.notesCount + 1);
    } catch (error) {
      console.error("Error posting note:", error);
    }
  }

  addNoteToDisplay(note) {
    const noteElement = $(`
            <div class="note-item mb-3">
                <div class="d-flex">
                    <div class="user-avatar mr-3">
                      <i
                        class="typcn typcn-user text-primary border-primary rounded-circle d-flex"
                        style="font-size: 48px; width: 3rem; height: 3rem"
                      ></i>
                    </div>
                    <div class="flex-grow-1">
                        <div class="note-header">
                            <strong>${note.user.name}</strong>
                            <small class="text-muted ml-2">
                                ${new Date(note.created_at).toLocaleString()}
                            </small>
                        </div>
                        <div class="note-content mt-1">
                            ${note.content}
                        </div>
                    </div>
                </div>
            </div>
        `);

    this.notesContainer.append(noteElement);
  }

  updateNotesCount(count) {
    this.notesCount = count;
    this.notesCountBtn.text(`Case Notes (${count})`);
  }
}

// Initialize the photo manager when the document is ready
document.addEventListener("DOMContentLoaded", () => {
  const photoManager = new PhotoManager();
});

document.addEventListener("DOMContentLoaded", function () {
  const caseId = document.getElementById("case-title").dataset.caseId;
  const notesManager = new NotesManager(caseId);

  // Handle Case Notes button click
  document.querySelector("#caseNotesBtn").addEventListener("click", () => {
    $("#notesModal").modal("show");
  });
});

// Update Case
const updateCase = async (details = {}) => {
  try {
    let uid = document.getElementById("case-title").dataset.uid;
    let response = await fetch("/api/cases/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...details, uid: uid }),
    });
    if (response.ok) {
      let data = await response.json();
      console.log(data);
    } else {
      console.log(await response.text());
      flashMessage(
        "There was an error. Refresh the page and try again.",
        "danger"
      );
    }
  } catch (error) {
    console.error(error);
    flashMessage(
      "Something went wrong. Please refresh the page and try again.",
      "danger"
    );
  }
};
// Update Image
const updateImage = async (el = null, data = null) => {
  console.log("Updating image");
  try {
    let payload;
    if (!data) {
      payload = {
        id: el.dataset.imageId,
        field: el.dataset.fieldname,
        value: el.value,
      };
    } else payload = data;
    let response = await fetch("/api/images/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (response.ok) {
      let data = await response.json();
      console.log(data);
    } else {
      console.log(await response.text());
      flashMessage(
        "There was an error. Refresh the page and try again.",
        "danger"
      );
    }
  } catch (error) {
    console.error(error);
    flashMessage(
      "Something went wrong. Please refresh the page and try again.",
      "danger"
    );
  }
};

document.querySelectorAll(".image-detail").forEach((field) => {
  field.addEventListener("blur", async () => await updateImage(field));
});

document.querySelectorAll(".date-field").forEach((field) => {
  const display = field.querySelector(".date-display");
  const input = field.querySelector(".date-input");

  // Show input on display click
  display.addEventListener("click", () => {
    display.classList.add("d-none");
    input.classList.remove("d-none");
    input.value =
      display.textContent.trim() === "NULL"
        ? new Date().toISOString().split("T")[0]
        : new Date(Date.parse(display.textContent.trim()))
            .toISOString()
            .split("T")[0];
    console.log(display.textContent.trim());
    console.log(input.value);
    input.focus();
  });

  // Hide input and update display when focus is lost
  input.addEventListener("blur", async () => {
    input.classList.add("d-none");
    display.classList.remove("d-none");

    // Format the date for display
    const date = new Date(input.value);
    display.textContent = date.toLocaleDateString("en-GB", {
      year: "numeric",
      month: "short",
      day: "2-digit",
    });
    if (date) await updateCase({ field: "dob", value: date });
  });

  // Handle Enter key
  input.addEventListener("keyup", (e) => {
    if (e.key === "Enter") {
      input.blur();
    }
  });
});

document.addEventListener("DOMContentLoaded", function () {
  const titleContainer = document.querySelector(".case-title-container");
  const titleDisplay = titleContainer.querySelector(".case-title");
  const titleInput = titleContainer.querySelector(".case-title-input");
  const editButton = titleContainer.querySelector(".edit-title-btn");

  function startEditing() {
    titleDisplay.classList.add("d-none");
    titleInput.classList.remove("d-none");
    titleInput.focus();
  }

  async function stopEditing() {
    titleInput.classList.add("d-none");
    titleDisplay.classList.remove("d-none");
    if (titleInput.value.trim() !== "") {
      titleDisplay.textContent = titleInput.value;
      await updateCase({ field: "name", value: titleInput.value });
    }
  }

  // Edit button click handler
  editButton.addEventListener("click", (e) => {
    e.preventDefault();
    startEditing();
  });

  // Handle input blur
  titleInput.addEventListener("blur", async () => {
    await stopEditing();
  });

  // Handle Enter key
  titleInput.addEventListener("keyup", async (e) => {
    if (e.key === "Enter") {
      await stopEditing();
    }
    // Escape key cancels editing
    else if (e.key === "Escape") {
      titleInput.value = titleDisplay.textContent;
      await stopEditing();
    }
  });

  // Gender Change Handler
  document
    .getElementById("genderSelect")
    .addEventListener("change", async (e) => {
      const newValue = e.target.value;
      console.log("Gender changed:", newValue);
      await updateCase({ field: "gender", value: newValue });
    });

  // Ethnicity Change Handler
  document
    .getElementById("ethnicitySelect")
    .addEventListener("change", async (e) => {
      const newValue = e.target.value;
      console.log("Ethnicity changed:", newValue);
      await updateCase({ field: "ethnicity", value: newValue });
    });

  // Measurements Change Handler
  document.querySelectorAll(".measurement").forEach((measurement) => {
    measurement.addEventListener("blur", async () => {
      let field = measurement.dataset.fieldname;
      let value = measurement.value;
      let details = { field, value };
      await updateCase(details);
    });
  });
});
