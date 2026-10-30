package sdfx.core.png;

import java.io.EOFException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

/** Bounded, read-only metadata presence probe. It never claims to preserve metadata. */
public final class ExternalImageMetadataProbe {
    public record Result(String format, Set<String> kinds, boolean complete) {
        public Result { kinds = Collections.unmodifiableSet(new LinkedHashSet<>(kinds)); }
    }
    private static final int MAX_PARTS = 65536;
    private ExternalImageMetadataProbe() {}

    public static Result inspect(Path path) throws IOException {
        try (RandomAccessFile f = new RandomAccessFile(path.toFile(), "r")) {
            long length = f.length();
            if (length < 2) return new Result("other", Set.of(), false);
            int a = f.readUnsignedByte(), b = f.readUnsignedByte();
            if (a == 0xff && b == 0xd8) return jpeg(f, length);
            f.seek(0);
            if (length >= 12 && ascii(f, 4).equals("RIFF")) {
                long riffSize = little32(f);
                if (!ascii(f, 4).equals("WEBP")) return new Result("other", Set.of(), false);
                if (riffSize < 4 || riffSize + 8 != length) throw new IOException("Invalid WebP RIFF length");
                return webp(f, length);
            }
            return new Result("other", Set.of(), false);
        }
    }

    private static Result jpeg(RandomAccessFile f, long length) throws IOException {
        Set<String> found = new LinkedHashSet<>();
        for (int count = 0; count < MAX_PARTS; count++) {
            if (f.getFilePointer() >= length || f.readUnsignedByte() != 0xff) throw new IOException("Invalid JPEG marker");
            int marker;
            do { marker = f.readUnsignedByte(); } while (marker == 0xff);
            if (marker == 0xda) return new Result("jpeg", found, false); // Later scans can contain metadata.
            if (marker == 0xd9) return new Result("jpeg", found, true);
            if (marker == 0x00 || marker == 0xd8) throw new IOException("Invalid JPEG marker");
            if (marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
            int size = f.readUnsignedShort();
            long start = f.getFilePointer(), end = start + size - 2L;
            if (size < 2 || end > length) throw new IOException("Invalid JPEG segment length");
            int prefixLength = (int)Math.min(size - 2L, 14);
            byte[] prefix = new byte[prefixLength];
            f.readFully(prefix);
            if (marker == 0xe0 && begins(prefix, "JFIF\0")) {
                if (size < 16) throw new IOException("Invalid JFIF header");
                found.add("density"); // Unit 0 still stores pixel aspect ratio.
            } else if (marker == 0xe1) {
                if (begins(prefix, "Exif\0\0")) found.add("exif");
                else found.add("other-app-metadata");
            } else if (marker == 0xe2 && begins(prefix, "ICC_PROFILE\0")) found.add("icc");
            else if (marker == 0xed) found.add("other-app-metadata");
            f.seek(end);
        }
        throw new IOException("Too many JPEG segments");
    }

    private static Result webp(RandomAccessFile f, long length) throws IOException {
        Set<String> found = new LinkedHashSet<>();
        int count = 0;
        while (f.getFilePointer() < length) {
            if (++count > MAX_PARTS || length - f.getFilePointer() < 8) throw new IOException("Invalid WebP chunk header");
            String kind = ascii(f, 4);
            long size = little32(f), start = f.getFilePointer(), end = start + size + (size & 1);
            if (end > length) throw new IOException("Invalid WebP chunk length");
            switch (kind) {
                case "ICCP" -> found.add("icc");
                case "EXIF" -> found.add("exif");
                case "XMP " -> found.add("xmp");
                case "VP8X" -> {
                    if (size != 10) throw new IOException("Invalid WebP extended header");
                    int flags = f.readUnsignedByte();
                    if ((flags & 0x20) != 0) found.add("icc");
                    if ((flags & 0x08) != 0) found.add("exif");
                    if ((flags & 0x04) != 0) found.add("xmp");
                }
                default -> { }
            }
            f.seek(end);
        }
        return new Result("webp", found, true);
    }
    private static boolean begins(byte[] bytes, String prefix) {
        byte[] wanted = prefix.getBytes(StandardCharsets.ISO_8859_1);
        if (bytes.length < wanted.length) return false;
        for (int i = 0; i < wanted.length; i++) if (bytes[i] != wanted[i]) return false;
        return true;
    }
    private static String ascii(RandomAccessFile f, int size) throws IOException {
        byte[] bytes = new byte[size]; f.readFully(bytes); return new String(bytes, StandardCharsets.ISO_8859_1);
    }
    private static long little32(RandomAccessFile f) throws IOException {
        return Integer.toUnsignedLong(Integer.reverseBytes(f.readInt()));
    }
}
