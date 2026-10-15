//
//  FrameDownsampler.swift
//  weard-it-go-ios
//

import CoreImage
import CoreImage.CIFilterBuiltins
import CoreVideo
import Foundation
import QuartzCore

/// A downsampled frame ready to publish.
nonisolated struct DownsampledFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let timeStampNs: Int64
    let width: Int
    let height: Int
    /// Set on roughly two frames a second, for the on-screen preview.
    let preview: CGImage?
}

/// Throttles incoming glasses frames to a target rate and scales them down into
/// BGRA pixel buffers on a private queue.
///
/// Called from the SDK's frame-delivery thread, so it never blocks: frames that
/// arrive before the next slot, or while the previous one is still being
/// scaled, are dropped rather than queued.
nonisolated final class FrameDownsampler: @unchecked Sendable {
    private let minInterval: CFTimeInterval
    private let maxLongEdge: CGFloat
    private let previewInterval: CFTimeInterval = 0.5

    private let lock = NSLock()
    private var nextSlot: CFTimeInterval = 0
    private var isBusy = false

    // Only touched on `queue`.
    private var pool: CVPixelBufferPool?
    private var poolSize: (width: Int, height: Int) = (0, 0)
    private var nextPreview: CFTimeInterval = 0

    private let queue = DispatchQueue(label: "weard-it-go.downsample", qos: .userInitiated)
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

    init(outputFrameRate: Double, maxLongEdge: Int) {
        minInterval = 1 / max(outputFrameRate, 0.01)
        self.maxLongEdge = CGFloat(maxLongEdge)
    }

    /// Offers a frame. `makeImage` is only invoked for frames that will be kept.
    func offer(makeImage: () -> CIImage?, completion: @escaping @Sendable (DownsampledFrame) -> Void) {
        let now = CACurrentMediaTime()
        lock.lock()
        guard now >= nextSlot, !isBusy else {
            lock.unlock()
            return
        }
        // Advance on a fixed grid so jitter doesn't lower the effective rate,
        // but never schedule into the past after a stall.
        nextSlot = max(nextSlot + minInterval, now + minInterval * 0.5)
        isBusy = true
        lock.unlock()

        guard let image = makeImage() else {
            finish()
            return
        }
        let timeStampNs = Int64(now * Double(NSEC_PER_SEC))
        queue.async { [self] in
            defer { finish() }
            guard let frame = scale(image, timeStampNs: timeStampNs, now: now) else { return }
            completion(frame)
        }
    }

    private func finish() {
        lock.lock()
        isBusy = false
        lock.unlock()
    }

    private func scale(_ image: CIImage, timeStampNs: Int64, now: CFTimeInterval) -> DownsampledFrame? {
        let extent = image.extent
        let longEdge = max(extent.width, extent.height)
        guard longEdge > 0 else { return nil }

        let scale = min(1, maxLongEdge / longEdge)
        // Video encoders want even dimensions.
        let width = Int(extent.width * scale) & ~1
        let height = Int(extent.height * scale) & ~1
        guard width >= 16, height >= 16 else { return nil }

        var output = image.transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
        if scale < 1 {
            // Lanczos avoids the aliasing a plain affine scale gives at large reductions.
            let filter = CIFilter.lanczosScaleTransform()
            filter.inputImage = output
            filter.scale = Float(scale)
            filter.aspectRatio = 1
            guard let scaled = filter.outputImage else { return nil }
            output = scaled
        }

        guard let buffer = makePixelBuffer(width: width, height: height) else { return nil }
        let bounds = CGRect(x: 0, y: 0, width: width, height: height)
        context.render(output, to: buffer, bounds: bounds, colorSpace: colorSpace)

        var preview: CGImage?
        if now >= nextPreview {
            nextPreview = now + previewInterval
            preview = context.createCGImage(CIImage(cvPixelBuffer: buffer), from: bounds)
        }
        return DownsampledFrame(pixelBuffer: buffer, timeStampNs: timeStampNs, width: width, height: height, preview: preview)
    }

    private func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        if pool == nil || poolSize != (width, height) {
            let attributes: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey: width,
                kCVPixelBufferHeightKey: height,
                kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
            ]
            var newPool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &newPool)
            pool = newPool
            poolSize = (width, height)
        }
        guard let pool else { return nil }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
        return buffer
    }
}
